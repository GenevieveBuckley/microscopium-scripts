import os
import glob
import sys
import re
import time
import click
import toolz as tz
from tqdm import tqdm
from skimage import io
from microscopium import preprocess as pre
from microscopium._util import generate_spiral

ROOT = '/scratch/su62/petermac/data'
os.chdir(ROOT)
OUT = os.path.join(ROOT, 'out')
os.makedirs(OUT)


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


############### illumination fields #####################
t0 = time.time()

input_images = sorted(glob.glob('*/*.TIF'))

exp = (r'(?P<dir>.*)/(?P<plate>mfgtmp_\d*)_'
       r'(?P<well>[A-P]\d\d)f(?P<field>\d\d)d(?P<channel>\d).TIF')

def field_channel(fn):
    match = re.match(exp, fn)
    if match is not None:
        return (match['field'], match['channel'])
    else:
        return 'none', 'none'

images_by_field = tz.groupby(field_channel, input_images)
illum_fields = {k: pre.find_background_illumination(fns, radius=41)
                for k, fns in tqdm(images_by_field.items(), 'illum')
                if k != ('none', 'none')}

for k, im in illum_fields.items():
    fn = 'illum-' + str.join('-', k) + '.png'
    io.imsave(im, os.path.join(OUT, fn))

t1 = time.time()
print(f'illumination estimated in {ftime(t1 - t0)}')

############### illumination correction #####################
for k, fns in tqdm(images_by_field.items(), 'corr'):
    corrected = pre.correct_multiimage_illumination(fns, illum_fields[k],
                                                    stretch_quantile=0.01)
    for fn, image in zip(fns, corrected):
        fnout = fn[:-4] + '.illum.png'
        io.imsave(image, fnout)

t2 = time.time()
print(f'illumination corrected in {ftime(t2 - t1)}')

############### montaging #####################

intermediate_images = sorted(glob.glob('*/*.illum.png'))
exp = (r'(?P<dir>.*)/(?P<plate>mfgtmp_\d*)_'
       r'(?P<well>[A-P]\d\d)f(?P<field>\d\d)d(?P<channel>\d).illum.png')

def plate_well(fn):
    match = re.match(exp, fn)
    if match in not None:
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

for i, (coord, fns) in enumerate(tqdm(by_coord.items(), 'montage')):
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


t3 = time.time()
print(f'montaged in {ftime(t3 - t2)}')


