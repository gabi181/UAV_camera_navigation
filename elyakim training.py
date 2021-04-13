import cv2 as cv
import matplotlib.pyplot as plt
from skimage.feature import hog
import  numpy as np
import random
import rotate_image

road_2020 = cv.imread('road_2020.png')          # queryImage
road_2018 = cv.imread('road_2018.png')          # queryImage

#configurations:
hog_alg = 0
hist_alg = 0
grey_alg = 0
thresh_alg = 0
blur_image = 1
rotate_im = 0
step_size = 13
window_size = 280
image_size = 120

if (hog_alg):
    fd, hog_road_2020 = hog(road_2020, orientations=12, pixels_per_cell=(8, 8),
                        cells_per_block=(8, 8), visualize=True, multichannel=True)
    road_2020 = np.array(hog_road_2020)
    fd, hog_road_2018 = hog(road_2018, orientations=12, pixels_per_cell=(8, 8),
                        cells_per_block=(8, 8), visualize=True, multichannel=True)
    road_2018 = np.array(hog_road_2018)
elif (grey_alg):
    road_2020 = cv.cvtColor(road_2020, cv.COLOR_BGR2GRAY)
    road_2018 = cv.cvtColor(road_2018, cv.COLOR_BGR2GRAY)
    if (thresh_alg):
        road_2020[road_2020>127] = 255
        road_2020[road_2020<=127] = 0
        road_2018[road_2018>127] = 255
        road_2018[road_2018<=127] = 0

if(blur_image):
    kernel = np.ones((10, 10), np.float32) / 100
    road_2020 = cv.filter2D(road_2020, -1, kernel)
    road_2018 = cv.filter2D(road_2018, -1, kernel)


#creat UVM image:
y_orig = random.randint(500,1400)
x_orig = random.randint(400,500)
image = road_2020[x_orig:x_orig+image_size,y_orig:y_orig+image_size]
if (rotate_im):
    image = rotate_image.rotate(image,30)
x_list = np.zeros(8)
y_list = np.zeros(8)
dist_list = np.zeros(8)
x_min = 0
y_min = 0
dist_min = 1000000000000000
for num in range(len(dist_list)):
    dist_list[num] = dist_min


if (hist_alg):
    image_r = np.histogram(image[:,:,0],20)
    image_g = np.histogram(image[:,:,1],20)
    image_b = np.histogram(image[:,:,2],20)
    image_hist = [image_r[0],image_g[0],image_b[0]]
for i in range(x_orig-window_size,x_orig+window_size,step_size):
    for j in range(y_orig-window_size,y_orig+window_size,step_size):
        image_to_compare = road_2018[i:i+image_size,j:j+image_size]
        if (hist_alg):
            image_to_compare_r = np.histogram(image_to_compare[:, :, 0],20)
            image_to_compare_g = np.histogram(image_to_compare[:, :, 1],20)
            image_to_compare_b = np.histogram(image_to_compare[:, :, 2],20)
            image_to_compare_hist = [image_to_compare_r[0], image_to_compare_g[0], image_to_compare_b[0]]
            dist = (np.sum((image_hist[0]-image_to_compare_hist[0])**2)+np.sum((image_hist[1]-image_to_compare_hist[1])**2)+np.sum((image_hist[2]-image_to_compare_hist[2])**2))/(3*image_size**2*256)
            if (dist<dist_min):
                x_list[np.argmax(dist_list)] = int(i)
                y_list[np.argmax(dist_list)] = int(j)
                dist_list[np.argmax(dist_list)] = dist
                dist_min = np.max(dist_list)
        else:
            dist = (np.sum((image-image_to_compare)**2))#/(image_size**2*256)
            if (dist < dist_min):
                dist_min = dist
                x_min = i
                y_min = j
i = 0
j = 0
print(x_list)
print(y_list)
print(dist_list)
if (hist_alg):
    dist_min = 10000000000000000
    for ind in range(len(x_list)):
        i = int(x_list[ind])
        j = int(y_list[ind])
        image_to_compare = road_2018[i:i+image_size,j:j+image_size]
        dist = (np.sum((image - image_to_compare) ** 2)) / (image_size ** 2 * 256)
        if(dist < dist_min):
            dist_min = dist
            x_min = i
            y_min = j

x_min = x_min+35
y_min = y_min+50

print(x_min,y_min)
print(x_orig,y_orig)
print(x_min-x_orig)
print(y_min-y_orig)

#show images:
road_2018[x_orig:x_orig+5,y_orig:y_orig+image_size,0] = 255
road_2018[x_orig+image_size:x_orig+image_size+5,y_orig:y_orig+image_size,0] = 255
road_2018[x_orig:x_orig+image_size,y_orig:y_orig+5,0] = 255
road_2018[x_orig:x_orig+image_size,y_orig+image_size:y_orig+image_size+5,0] = 255
road_2020[x_min:x_min+5,y_min:y_min+image_size,0] = 255
road_2020[x_min+image_size:x_min+image_size+5,y_min:y_min+image_size,0] = 255
road_2020[x_min:x_min+image_size,y_min:y_min+5,0] = 255
road_2020[x_min:x_min+image_size,y_min+image_size:y_min+image_size+5,0] = 255
road_2018[x_orig-window_size:x_orig-window_size+5,y_orig-window_size:y_orig+window_size+image_size,1] = 255
road_2018[x_orig+window_size+image_size:x_orig+window_size+image_size+5,y_orig-window_size:y_orig+window_size+image_size,1] = 255
road_2018[x_orig-window_size:x_orig+window_size+image_size,y_orig-window_size:y_orig-window_size+5,1] = 255
road_2018[x_orig-window_size:x_orig+window_size+image_size,y_orig+window_size+image_size:y_orig+window_size+image_size+5,1] = 255
road_2020[x_min-window_size:x_min-window_size+5,y_min-window_size:y_min+window_size+image_size,1] = 255
road_2020[x_min+window_size+image_size:x_min+window_size+image_size+5,y_min-window_size:y_min+window_size+image_size,1] = 255
road_2020[x_min-window_size:x_min+window_size+image_size,y_min-window_size:y_min-window_size+5,1] = 255
road_2020[x_min-window_size:x_min+window_size+image_size,y_min+window_size+image_size:y_min+window_size+image_size+5,1] = 255
plt.figure(1)
plt.subplot(2,1,1)
plt.imshow(road_2018)
plt.subplot(2,1,2)
plt.imshow(road_2020)
#plt.figure(3)
#plt.imshow(road_2020)
plt.show()
