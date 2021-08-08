import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature



def change_resolution(image,resize_value):
    resize_len = resize_value
    resize_width = resize_value
    image_len = len(image[0, :])
    image_width = len(image[:, 0])
    size = int(image_len/resize_len) , int(image_width/resize_width)
    image_resized = cv.resize(image, size)
    return image_resized

def add_noise(image):
    # Generate noisy image of a square
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = ndi.rotate(image, 0, mode='constant')
    image = ndi.gaussian_filter(image, 4)
    image = random_noise(image, mode='speckle', mean=0.1)
    return image

## Compute the Canny filter for two values of sigma
#edges1 = feature.canny(image)
#edges2 = feature.canny(image, sigma=3)
#
## display results
#fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
#
#ax[0].imshow(image, cmap='gray')
#ax[0].set_title('noisy image', fontsize=20)
#
#ax[1].imshow(edges1, cmap='gray')
#ax[1].set_title(r'Canny filter, $\sigma=1$', fontsize=20)
#
#ax[2].imshow(edges2, cmap='gray')
#ax[2].set_title(r'Canny filter, $\sigma=3$', fontsize=20)
#
#for a in ax:
#    a.axis('off')
#
#fig.tight_layout()
#plt.show()

#image_path_2020 = "road_2020.png"
#image_path_2018 = "road_2018.png"
#image_2020 = np.array(cv.imread(image_path_2020))
#image_2018 = np.array(cv.imread(image_path_2018))
#
#noisy_image_2020 = add_noise(image_2020)
#noisy_image_2018 = add_noise(image_2018)
#
#resize_after_noise_2020 =change_resolution(add_noise(image_2020),2)
#plt.figure(0)
#plt.imshow(noisy_image_2020,cmap='gray')
#plt.figure(1)
#plt.imshow(resize_after_noise_2020,cmap='gray')
#plt.tight_layout()
#plt.show()
