import os
import itertools
import glob
import sys
import re
import time
import click
import toolz as tz
import numpy as np
from dask_jobqueue import SLURMCluster as Cluster
from dask import delayed
from dask.distributed import Client
from tqdm import tqdm
from skimage import io
import pandas as pd
from microscopium import features as feat


ROOT = '/scratch/su62/petermac/data'
OUT = os.path.join(ROOT, 'out')
os.chdir(OUT)


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


############### feature computation #####################

# config in /home/jnun0003/.config/dask/jobqueue.yaml
cluster = Cluster(walltime='01:00:00', cores=1, memory='4gb', job_cpu=1,
                  job_mem='4gb')
cluster.adapt(minimum=0, maximum=48, target_duration='4h')

t3 = time.time()

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


def features(fn):
    return features_with_names(fn)[0]


feature_names = features_with_names(filenames[0])[1]

client = Client(cluster)

feature_futures = client.map(features, filenames)
X = np.empty((len(filenames), len(feature_names)), dtype=np.float32)
for i, future in tqdm(enumerate(feature_futures), 'features'):
    X[i, :] = future.result()

t4 = time.time()
print(f'features computed in {ftime(t4 - t3)}')

features = pd.DataFrame(X, columns=feature_names)
features['index'] = list(map('-'.join, plate_wells))
features['filenames'] = filenames

features.to_csv('data.csv')
