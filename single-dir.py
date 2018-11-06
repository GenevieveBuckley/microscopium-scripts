import os
import sys
import re
import click
import toolz as tz
from skimage import io
from microscopium import preprocess as pre
from microscopium._util import generate_spiral

ROOT = '/scratch/su62/petermac/data'
OUT = os.path.join(ROOT, 'out')

os.makedirs(OUT)


def get_surrounding_dirs(indir, root=None, radius=3):
    root = root or ROOT
    dirs = list(filter(os.path.isdir, os.listdir(root)))
    idx = dirs.index(indir)
    low = min(0, idx - radius)
    high = max(idx + radius, len(dirs) - 1)
    return dirs[low:high]


def get_full_paths(dirs, ext):
    fns = []
    for d in sorted(dirs):
        for fn in sorted(os.listdir(d)):
            if fn.endswith(ext):
                fns.append(os.path.join(d, fn))
    return fns


@click.command()
@click.argument('indir', help='input directory')
@click.option('--outdir', help='output directory')
def main(indir, outdir=None):
    indir = indir.rstrip('/')  # ensure no trailing slash
    stub = os.path.split(indir)[1]  # we'll use this for output
    outdir = outdir or os.path.join(OUT, stub)
    exp = (r'(?P<dir>.*)/(?P<plate>mfgtmp_\d*)_'
           r'(?P<well>[A-P]\d\d)f(?P<field>\d\d)d(?P<channel>\d).TIF')
    files = [os.path.join(indir, fn) for fn in sorted(os.listdir(indir))]
    illum_files = get_surrounding_dirs(indir)
    def well_channel(fn):
        match = re.match(exp, fn)
        if match is not None:
            return (match['well'], match['channel'])
        else:
            return None
    illum_groups = tz.groupby(well_channel, illum_files)
    for (well, ch), files in illum_groups.items():
        illum = pre.find_background_illumination(illum_files, quantile=0.05,
                                                 stretch_quantile=0.001)
        io.imsave(str.join('_', ['illum', well, ch]) + '.tif', illum)
        pre.correct_multiimage_illumination(
    montage_order = generate_spiral((5, 5), 'right', clockwise=False)


if __name__ == '__main__':
    main()
