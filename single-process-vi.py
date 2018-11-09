import os
import itertools
import glob
import sys
import re
import time
import click
import toolz as tz
from dask_jobqueue import SLURMCluster as Cluster
from dask import delayed
from dask.distributed import Client
from tqdm import tqdm
from skimage import io
from microscopium import preprocess as pre
from microscopium._util import generate_spiral

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
    field, ch = key
    fns = glob.glob(f'*/*f{field}d{ch}.TIF')
    illum = pre.find_background_illumination(fns, radius=41)
    return illum

# set up dask Futures to be computed on cluster (client)
keys = list(itertools.product((f'{f:02}' for f in range(25)),
                              ['0', '1']))
client = Client(cluster)
illum_fields = {k: client.submit(find_background, k) for k in keys}

# now gather the results
for k in tqdm(illum_fields, 'illum'):
    illum_fields[k] = illum_fields[k].result()
    im = illum_fields[k]
    fn = 'illum-' + str.join('-', k) + '.png'
    io.imsave(im, os.path.join(OUT, fn))

t1 = time.time()
print(f'illumination estimated in {ftime(t1 - t0)}')

############### illumination correction #####################
# set up dask Futures to be computed on cluster (client)
def correct_illumination(key, illum_field):
    print(f'fixing background for {key}')
    field, ch = key
    fns = glob.glob(f'*/*f{field}d{ch}.TIF')
    corrected = pre.correct_multiimage_illumination(fns, illum_field,
                                                    stretch_quantile=0.01)
    for fn, image in zip(fns, corrected):
        fnout = fn[:-4] + '.illum.png'
        io.imsave(image, fnout)
    return 'done'

for k in keys:
    client.submit(correct_illumination, k, illum_fields[k])

for k in tqdm(keys, 'illum-corr'):
    client.result()

t2 = time.time()
print(f'illumination corrected in {ftime(t2 - t1)}')

############### montaging #####################

# these are much shorter tasks, so we start a new cluster:
cluster = Cluster(walltime='00:15:00')
cluster.adapt(minimum=2, maximum=16, target_duration='12h')


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

by_coord = tz.groupby(intermediate_images, plate_well)

order = generate_spiral((5, 5), 'right', clockwise=False) 

def montage(coord, fns):
    by_ch = tz.groupby(ch, fns)
    montaged = {}
    for ch, fns_ch in sorted(by_ch.items()):
        montaged[ch] = pre.montage_with_missing(fns_ch, order=order,
                                                re_string=exp,
                                                re_group='field')[0]
    stacked = pre.stack_channels(montaged.values(), order=[None, 1, 0])
    plate, well = coord
    fn_out = os.path.join(OUT, plate, '-'.join(coord)) + '.jpg'
    io.imsave(fn_out, stacked, quality=95)


client = Client(cluster)
results = [client.submit(montage, coord, fns)
           for coord, fns in by_coord.items()]

for r in tqdm(results, 'montage'):
    r.result()

t3 = time.time()
print(f'montaged in {ftime(t3 - t2)}')

