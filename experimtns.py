import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog

hist_exp =0
grey_exp =0
hog_exp =1
blur_image = 0

image_new = np.array(cv.imread('exp_new.png'))  # queryImage
image_old = np.array(cv.imread('exp_old.png'))  # queryImage
image_old_r_mean = image_old - np.mean(image_old[:,:,0]) + np.mean(image_new[:,:,0])
image_new_grey = cv.cvtColor(image_new, cv.COLOR_BGR2GRAY)
image_old_grey = cv.cvtColor(image_old, cv.COLOR_BGR2GRAY)

hist_r_new,bin_r_new = np.histogram(image_new[:,:,0],10)
hist_r_old,bin_r_old = np.histogram(image_old[:,:,0],10)
hist_r_old_mean,bin_r_old_mean = np.histogram(image_old_r_mean,10)
hist_new_grey,bin_new_grey = np.histogram(image_new_grey,10)
hist_old_grey,bin_old_grey = np.histogram(image_old_grey,10)


fd, image_new_hog = hog(image_new_grey, orientations=12, pixels_per_cell=(2, 2),
                    cells_per_block=(8, 8), visualize=True, multichannel=False)
fd, image_old_hog = hog(image_old_grey, orientations=12, pixels_per_cell=(2, 2),
                    cells_per_block=(8, 8), visualize=True, multichannel=False)
image_new_hog = image_new_hog[1::2,1::2]
image_old_hog = image_old_hog[1::2,1::2]


image_new_hog = (image_new_hog>(np.mean(image_new_hog)+5))
image_old_hog = (image_old_hog>(np.mean(image_old_hog)+5))

if (blur_image):
    kernel = np.ones((3, 3), np.float32) / 9
    image_new_hog = cv.filter2D(image_new_hog, -1, kernel)
    image_old_hog = cv.filter2D(image_old_hog, -1, kernel)


if (hist_exp):
    plt.figure(1)
    plt.bar(bin_r_new[:-1],hist_r_new,width=15)
    #plt.hist(hist_r_new[0])
    plt.figure(2)
    plt.bar(bin_r_old[:-1],hist_r_old,width=15)
    plt.figure(3)
    plt.bar(bin_r_old_mean[:-1],hist_r_old_mean,width=15)
    plt.figure(4)
    plt.imshow(image_new_grey)
    #plt.hist(hist_r_old[0])
    #plt.figure(3)
    #plt.imshow(image_new)

if (grey_exp):
    plt.figure(1)
    plt.bar(bin_new_grey[:-1], hist_new_grey, width=15)
    plt.figure(2)
    plt.bar(bin_old_grey[:-1], hist_old_grey, width=15)

if (hog_exp):
    plt.figure(1)
    plt.imshow(image_new_hog)
    plt.figure(2)
    plt.imshow(image_old_hog)
plt.show()
