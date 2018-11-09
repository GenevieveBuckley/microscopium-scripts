import os
import sys
import glob
from skimage import io

d = os.path.join('/scratch/su62/petermac/data', sys.argv[1])
os.chdir(d)
files = glob.glob('*.TIF')
for f in files:
    try:
        image = io.imread(f)
    except ValueError:
        print(f'error reading file {f}')
        os.rename(f, f + '.bad')
        continue
    if image.shape == (512, 512, 3):
        io.imsave(f, image[..., 0], compress=1)
