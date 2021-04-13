# import gdal
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from sim_func import *
from scipy import ndimage
from pathlib import Path
import rotate_image
import matching
from skimage.feature import hog

p = Path('.')
#raster_path = p.resolve().parent / 'data' / 'ecw1.tif'
raster_path_2020 = "wetransfer-e55797 2020/ecw1.tif"
raster_path_2018 = "wetransfer-8334c6 2018/TechnionOrtho20181.tif"

im = tifffile.imread(raster_path_2020)[:][:][1:]
im = np.array(im)
im_18 = tifffile.imread(raster_path_2018)[:][:][1:]
im_18 = np.array(im_18)
#%%
road_im = return_road_image(im)
road_im_18 = return_road_image(im_18)
plt.figure(1)
plt.imshow(road_im)
plt.figure(2)
plt.imshow(road_im_18)
plt.show()

fd, hog_image = hog(road_im, orientations=12, pixels_per_cell=(8, 8),
                    cells_per_block=(8, 8), visualize=True, multichannel=True)
for i in range(10):

    images = create_images(hog_image)

    small_im = rotate_image.rotate(images[1],60)
    med_im = images[0]

    match_cors = matching.match_with_sift(med_im,small_im)

    if(match_cors==[False,False]):
        print("unsocsseed to match")
        continue

    print('macthed cordinates are:',[match_cors[0]+images[2][0]-200,match_cors[1]+images[2][1]-200])
    print('original cordinates are:',images[3])

#plt.figure(1)
#plt.imshow(small_im)
#plt.figure(2)
#plt.imshow(med_im)
#plt.show()
#print('hi')
#%%

