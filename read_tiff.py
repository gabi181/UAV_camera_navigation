# import gdal
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from sim_func import *
from scipy import ndimage
from pathlib import Path

p = Path('.')
raster_path = p.resolve().parent / 'data' / 'ecw1.tif'

im = tifffile.imread(raster_path)[:][:][1:]
im = np.array(im)
#%%
road_im = return_road_image(im)
images = create_images(road_im)
small_im = images[1]
#small_im_rotated = ndimage.rotate(small_im, 45)
big_im = images[0]


plt.figure(1)
plt.imshow(small_im)
plt.figure(2)
plt.imshow(big_im)
plt.show()

#%%

