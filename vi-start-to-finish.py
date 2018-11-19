"""Use dask-jobqueue to correct illumination, montage, and extract features."""
import os
import itertools
import glob
import sys
import re
import time
import click
import toolz as tz
import numpy as np
import pandas as pd
from dask_jobqueue import SLURMCluster as Cluster
from dask import delayed
from dask.distributed import Client, as_completed
from distributed.scheduler import KilledWorker
from tqdm import tqdm
from skimage import io
from microscopium import preprocess as pre
from microscopium._util import generate_spiral
from microscopium import features as feat


ROOT = '/scratch/su62/petermac/data'
os.chdir(ROOT)
OUT = os.path.join(ROOT, 'out')
os.makedirs(OUT, exist_ok=True)


def ftime(seconds):
    string = ''
    units = ['d', 'h', 'm', 's']
    values = [3600 * 24, 3600, 60, 1]
    for unit, value in zip(units, values):
        amount = seconds // value
        if amount > 0:
            string += f'{amount}{unit}'
        seconds = seconds % value
    return string

# config in /home/jnun0003/.config/dask/jobqueue.yaml
cluster = Cluster(walltime='06:00:00')
cluster.adapt(minimum=2, maximum=16, target_duration='1d')

############### illumination fields #####################
t0 = time.time()

exp = (r'(?P<dir>.*)/(?P<plate>mfgtmp_\d*)_'
       r'(?P<well>[A-P]\d\d)f(?P<field>\d\d)d(?P<channel>\d).TIF')

def find_background(key):
    print(f'finding background for {key}')
    plate, ch = key
    fns = glob.glob(f'{plate}/{plate}_*d{ch}.TIF')
    illum = pre.find_background_illumination(fns, radius=151)
    return illum

# set up dask Futures to be computed on cluster (client)
dirs = filter(lambda x: x.startswith('mfgtmp'), os.listdir('.'))
keys = list(itertools.product(dirs, ['0', '1']))
client = Client(cluster)
illum_fields = {k: client.submit(find_background, k) for k in keys}

# now gather the results
for k in tqdm(illum_fields, 'illum'):
    future = illum_fields[k]
    illum_fields[k] = future.result()
    future.cancel()
    im = illum_fields[k]
    fn = 'illum-' + str.join('-', k) + '.png'
    io.imsave(os.path.join(OUT, fn), im)

t1 = time.time()
print(f'illumination estimated in {ftime(t1 - t0)}')

############### illumination correction #####################
# set up dask Futures to be computed on cluster (client)
def correct_illumination(tup):
    key, illum_field = tup
    print(f'fixing background for {key}')
    plate, ch = key
    fns = glob.glob(f'{plate}/{plate}_*d{ch}.TIF')
    corrected = pre.correct_multiimage_illumination(fns, illum_field,
                                                    stretch_quantile=0.01)
    for fn, image in zip(fns, corrected):
        fnout = fn[:-4] + '.illum.png'
        io.imsave(fnout, image)
    return key


futures = client.map(correct_illumination, illum_fields.items())

for fut in tqdm(as_completed(futures), desc='illum-corr', total=len(keys)):
    r = fut.result()
    fut.cancel()

t2 = time.time()
print(f'illumination corrected in {ftime(t2 - t1)}')

############### montaging #####################

# these are much shorter tasks, so we start a new cluster:
cluster.close()
cluster = Cluster(walltime='01:00:00', cores=1, memory='4gb', job_cpu=1, job_mem='4gb')
cluster.adapt(minimum=0, maximum=48, target_duration='4h')


intermediate_images = sorted(glob.glob('*/*.illum.png'))
exp = (r'(?P<dir>.*)/(?P<plate>mfgtmp_\d*)_'
       r'(?P<well>[A-P]\d\d)f(?P<field>\d\d)d(?P<channel>\d).illum.png')

def plate_well(fn):
    match = re.match(exp, fn)
    if match is not None:
        return match['plate'], match['well']
    else:
        return 'none', 'none'

def ch(fn):
    match = re.match(exp, fn)
    if match is not None:
        return int(match['channel'])
    else:
        return 'none'

by_coord = tz.groupby(plate_well, intermediate_images)

order = generate_spiral((5, 5), 'right', clockwise=False) 

def montage(coord, fns):
    plate, well = coord
    fn_out = os.path.join(OUT, plate, '-'.join(coord)) + '.jpg'
    # uncomment line below if starting pipeline halfway
    #if os.path.exists(fn_out):
    #    return
    by_ch = tz.groupby(ch, fns)
    montaged = {}
    for c, fns_ch in sorted(by_ch.items()):
        montaged[c] = pre.montage_with_missing(fns_ch, order=order,
                                                re_string=exp,
                                                re_group='field')[0]
    stacked = pre.stack_channels(montaged.values(), order=[None, 1, 0])
    io.imsave(fn_out, stacked, quality=95)


plates = set(coord[0] for coord in by_coord)
for plate in plates:
    os.makedirs(os.path.join(OUT, plate), exist_ok=True)


client = Client(cluster)
results = [client.submit(montage, coord, fns)
           for coord, fns in by_coord.items()]

for r in tqdm(as_completed(results), desc='montage', total=len(by_coord)):
    r.cancel()

t3 = time.time()
print(f'montaged in {ftime(t3 - t2)}')





############### feature computation #####################

os.chdir(OUT)

plates = [d for d in os.listdir('.')
          if os.path.isdir(d) and d.startswith('mfgtmp')]

rows = [chr(n + ord('A')) for n in range(16)]
cols = [f'{i:02}' for i in range(1, 25)]
wells = [''.join([row, col]) for row in rows for col in cols]

all_plate_wells = list(itertools.product(plates, wells))
all_filenames = [os.path.join(plate, '-'.join([plate, well]) + '.jpg')
                 for plate, well in all_plate_wells]
plate_wells, filenames = zip(*[(plate_well, filename)
                               for plate_well, filename
                               in zip(all_plate_wells, all_filenames)
                               if os.path.exists(filename)])


def features_with_names(fn):
    image = io.imread(fn)
    fvector, names = feat.default_feature_map(image, channels=[2, 1],
                                              channel_names=['dapi', 'polyA'],
                                              sample_size=1000)
    return fvector, names


def features(num_fn):
    num, fn = num_fn
    return num, features_with_names(fn)[0]


feature_names = features_with_names(filenames[0])[1]

client = Client(cluster)

X = np.empty((len(filenames), len(feature_names)), dtype=np.float32)
with tqdm(total=len(X), desc='features') as progress:
    queue = as_completed(client.map(features, enumerate(filenames)))
    for future in queue:
        try:
            i, v = future.result()
        except KilledWorker:
            client.retry([future])
            queue.add(future)
        else:
            X[i, :] = v
            future.cancel()
            progress.update(1)

t4 = time.time()
print(f'features computed in {ftime(t4 - t3)}')

features = pd.DataFrame(X, columns=feature_names)
features['index'] = list(map('-'.join, plate_wells))
features['filenames'] = filenames

features.to_csv('data.csv')

