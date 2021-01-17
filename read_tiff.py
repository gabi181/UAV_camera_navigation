import gdal
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import sim_func
from scipy import ndimage

raster_path = "wetransfer-e55797/ecw1.tif"
im = tifffile.imread(raster_path)[:][:][1:]
im = np.array(im)

images = sim_func.create_images(im)
small_im = images[1]
small_im_rotated = ndimage.rotate(small_im, 45)
big_im = images[0]

plt.figure(1)
plt.imshow(small_im)
plt.figure(2)
plt.imshow(small_im_rotated)
#plt.show()
plt.figure(3)
plt.imshow(big_im)
plt.show()


#raster_dataset = gdal.OpenEx(raster_path, gdal.GA_ReadOnly)
#geo_transform = raster_dataset.GetGeoTransform()
#proj = raster_dataset.GetProjectionRef()

#bands_data = []
#for b in range(1, raster_dataset.RasterCount+1):
#    band = raster_dataset.GetRasterBand(b)
#    bands_data.append(band.ReadAsArray())

#im = gdal.Open(raster_path).ReadAsArray()
#print(im.shape)

#rgb = np.zeros((im.shape[1],im.shape[2], 3))
#rgb[:,:,0] = im[1]
#rgb[:,:,1] = im[2]
#rgb[:,:,2] = im[3] #see secondincluded image for output


#plt.imshow((rgb * 255).astype(np.uint8))
#plt.show()