# import gdal
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from sim_func import *
from scipy import ndimage
from pathlib import Path
import rotate_image
import matching

p = Path('.')
#raster_path = p.resolve().parent / 'data' / 'ecw1.tif'
raster_path_elyakim = "wetransfer-e55797/ecw1.tif"

im = tifffile.imread(raster_path_elyakim)[:][:][1:]
im = np.array(im)
#%%
road_im = return_road_image(im)
images = create_images(road_im)

small_im = rotate_image.rotate(images[1],30)
med_im = images[0]

matching.match_with_sift(med_im,small_im)

plt.figure(1)
plt.imshow(small_im)
plt.figure(2)
plt.imshow(med_im)
plt.show()
#print('hi')
#%%

