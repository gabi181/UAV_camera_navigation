import cv2 as cv
import matplotlib.pyplot as plt
from skimage.feature import hog
import  numpy as np
import random
import rotate_image
import tifffile
import sim_func
import func_for_pf_alg

#configurations:
same_year = 1
road_alg = 0
hog_alg = 0
hist_alg = 0
ulman_alg = 0
mean_hist = 0
max_color_alg = 0
max_span = 5
grey_alg = 0
thresh_alg = 0
blur_image = 0
rotate_im = 0
step_size = 15
hist_col = 30
hist_list_len = 50
window_size = 600
image_size = 200

# generate large image
road_2020_large,road_2018_large = func_for_pf_alg.generate_large_image(road_alg=road_alg,ulman_alg=ulman_alg,same_year=same_year)

road_2020 = 0
road_2018 = 0
for nn in range(1):
   # generate mid image
   road_2020,road_2018 = func_for_pf_alg.generate_mid_image(road_2018_large,road_2020_large,window_size,window_size,road_alg=road_alg,blur_image=blur_image,ulman_alg=ulman_alg)

   # generate small image
   image,x_orig,y_orig = func_for_pf_alg.generate_uvm_image(road_2020, image_size, image_size, rotate_im, road_alg=road_alg,ulman_alg=ulman_alg)

   #plt.figure(3)
   #plt.imshow(image)
   #plt.show()
   # clculate diff function
   x_min,y_min,diff_mat = func_for_pf_alg.calc_im_diff(image,road_2018,step_size,hist_alg,hist_col,hist_list_len,max_color_alg=max_color_alg,max_span=max_span,hog_alg=hog_alg,mean_hist=mean_hist)

   print(x_min,y_min)
   print(x_orig,y_orig)
   print(x_min-x_orig)
   print(y_min-y_orig)

   #show images:
   #road_2018[x_orig:x_orig+5,y_orig:y_orig+image_size,0] = 255
   #road_2018[x_orig+image_size:x_orig+image_size+5,y_orig:y_orig+image_size,0] = 255
   #road_2018[x_orig:x_orig+image_size,y_orig:y_orig+5,0] = 255
   #road_2018[x_orig:x_orig+image_size,y_orig+image_size:y_orig+image_size+5,0] = 255
   #road_2020[x_min:x_min+5,y_min:y_min+image_size,0] = 255
   #road_2020[x_min+image_size:x_min+image_size+5,y_min:y_min+image_size,0] = 255
   #road_2020[x_min:x_min+image_size,y_min:y_min+5,0] = 255
   #road_2020[x_min:x_min+image_size,y_min+image_size:y_min+image_size+5,0] = 255
   #road_2018[x_orig-window_size:x_orig-window_size+5,y_orig-window_size:y_orig+window_size+image_size,1] = 255
   #road_2018[x_orig+window_size+image_size:x_orig+window_size+image_size+5,y_orig-window_size:y_orig+window_size+image_size,1] = 255
   #road_2018[x_orig-window_size:x_orig+window_size+image_size,y_orig-window_size:y_orig-window_size+5,1] = 255
   #road_2018[x_orig-window_size:x_orig+window_size+image_size,y_orig+window_size+image_size:y_orig+window_size+image_size+5,1] = 255
   #road_2020[x_min-window_size:x_min-window_size+5,y_min-window_size:y_min+window_size+image_size,1] = 255
   #road_2020[x_min+window_size+image_size:x_min+window_size+image_size+5,y_min-window_size:y_min+window_size+image_size,1] = 255
   #road_2020[x_min-window_size:x_min+window_size+image_size,y_min-window_size:y_min-window_size+5,1] = 255
   #road_2020[x_min-window_size:x_min+window_size+image_size,y_min+window_size+image_size:y_min+window_size+image_size+5,1] = 255
   orig_im = func_for_pf_alg.generate_im_to_show(road_2018,x_orig,y_orig,image_size,image_size)
   predict_im = func_for_pf_alg.generate_im_to_show(road_2020, x_min, y_min, image_size, image_size)

   #plt.figure(nn)
   plt.subplot(2,1,1)
   plt.imshow(orig_im)
   plt.subplot(2,1,2)
   plt.imshow(predict_im)
   #plt.subplot(3,1,3)
   #if(ulman_alg==0):
   #    plt.figure(2*nn+1)
   #    plt.matshow(diff_mat)
   #plt.figure(3)
   #plt.imshow(road_2020)

   plt.show()
   plt.clf()

