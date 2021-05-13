import numpy as np
import cv2 as cv
import tifffile
import sim_func
import random
import rotate_image
import matplotlib.pyplot as plt
from skimage.feature import hog

def calc_im_diff(image, mid_image, step_size=17, hist_alg=1, hist_col=40, hist_list_len=1, same_year=0,mean_hist=1,max_color_alg=0,max_span=5,hog_alg=0):
    image_len = len(image)
    image_width = len(image[0])
    mid_image_len = len(mid_image)
    mid_image_width = len(mid_image[0])
    x_list = np.zeros(hist_list_len)
    y_list = np.zeros(hist_list_len)
    dist_list = np.zeros(hist_list_len)
    x_min = 0
    y_min = 0
    dist_min = 1000000000000000
    for num in range(len(dist_list)):
        dist_list[num] = dist_min

    if (hog_alg):
        image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        mid_image_grey = cv.cvtColor(mid_image, cv.COLOR_BGR2GRAY)
        fd, image_grey = hog(image_grey, orientations=12, pixels_per_cell=(2, 2),
                                cells_per_block=(8, 8), visualize=True, multichannel=False)
        fd, mid_image_grey = hog(mid_image_grey, orientations=12, pixels_per_cell=(2, 2),
                                cells_per_block=(8, 8), visualize=True, multichannel=False)
        image = image_grey[1::2, 1::2]
        mid_image = mid_image_grey[1::2, 1::2]
        image_len = len(image)
        image_width = len(image[0])
        mid_image_len = len(mid_image)
        mid_image_width = len(mid_image[0])
        plt.figure(2)
        plt.imshow(mid_image)
        plt.figure(3)
        plt.imshow(image)
        plt.show()

    i_diff = 0
    j_diff = 0
    diff_mat = np.zeros([len(range(0,mid_image_len-image_len , step_size)),len(range(0, mid_image_width - image_width, step_size))])

    if (hist_alg):
        image_r = np.histogram(image[:, :, 0], hist_col)
        image_g = np.histogram(image[:, :, 1], hist_col)
        image_b = np.histogram(image[:, :, 2], hist_col)
        print("elyakim" + str(image_r[0]))
        #mean_r = np.mean(image_r[0])
        #mean_g = np.mean(image_g[0])
        #mean_b = np.mean(image_b[0])
        mean_r = np.mean(image[:, :, 0])
        mean_g = np.mean(image[:, :, 1])
        mean_b = np.mean(image[:, :, 2])
        image_hist = [image_r[0], image_g[0], image_b[0]]
    for i in range(0,mid_image_len-image_len , step_size):
        for j in range(0, mid_image_width - image_width, step_size):
            image_to_compare = mid_image[i:i + image_len, j:j + image_width]
            if (mean_hist):
                mean_r_to_compere = np.mean(image_to_compare[:,:,0])
                mean_g_to_compere = np.mean(image_to_compare[:,:,1])
                mean_b_to_compere = np.mean(image_to_compare[:,:,2])
                image_to_compare_r = image_to_compare[:, :, 0] + (-mean_r_to_compere + mean_r)
                image_to_compare_g = image_to_compare[:, :, 1] + (-mean_g_to_compere + mean_g)
                image_to_compare_b = image_to_compare[:, :, 2] + (-mean_b_to_compere + mean_b)
            if (hist_alg):
                image_to_compare_r = np.histogram(image_to_compare[:,:,0], hist_col)
                image_to_compare_g = np.histogram(image_to_compare[:,:,1], hist_col)
                image_to_compare_b = np.histogram(image_to_compare[:,:,2], hist_col)

                #mean_r_to_compere = np.mean(image_to_compare_r[0])
                #mean_g_to_compere = np.mean(image_to_compare_g[0])
                #mean_b_to_compere = np.mean(image_to_compare_b[0])
                #if (mean_hist):
                #    image_to_compare_r = image_to_compare_r - mean_r_to_compere + mean_r
                #    image_to_compare_g = image_to_compare_g - mean_g_to_compere + mean_g
                #    image_to_compare_b = image_to_compare_b - mean_b_to_compere + mean_b
                image_to_compare_hist = [image_to_compare_r[0], image_to_compare_g[0], image_to_compare_b[0]]
                if (max_color_alg):
                    image_r_ind = np.argmax(image_hist[0])
                    image_g_ind = np.argmax(image_hist[1])
                    image_b_ind = np.argmax(image_hist[2])
                    image_to_compare_r_ind = np.argmax(image_to_compare_hist[0])
                    image_to_compare_g_ind = np.argmax(image_to_compare_hist[1])
                    image_to_compare_b_ind = np.argmax(image_to_compare_hist[2])
                    dist = (np.sum((image_hist[0][image_r_ind-max_span:image_r_ind+max_span] - image_to_compare_hist[0][image_to_compare_r_ind-max_span:image_to_compare_r_ind+max_span]) ** 2) + np.sum(
                        (image_hist[1][image_g_ind-max_span:image_g_ind+max_span] - image_to_compare_hist[1][image_to_compare_g_ind-max_span:image_to_compare_g_ind+max_span]) ** 2) + np.sum(
                        (image_hist[2][image_b_ind-max_span:image_b_ind+max_span] - image_to_compare_hist[2][image_to_compare_b_ind-max_span:image_to_compare_b_ind+max_span]) ** 2)) / (3 * image_len * image_width * 256)
                else:
                    dist = (np.sum((image_hist[0] - image_to_compare_hist[0]) ** 2) + np.sum(
                        (image_hist[1] - image_to_compare_hist[1]) ** 2) + np.sum(
                        (image_hist[2] - image_to_compare_hist[2]) ** 2)) / (3 * image_len * image_width * 256)
                if (dist < dist_min):
                    x_list[np.argmax(dist_list)] = int(i)
                    y_list[np.argmax(dist_list)] = int(j)
                    dist_list[np.argmax(dist_list)] = dist
                    dist_min = np.max(dist_list)
            else:
                dist = (np.sum((image - image_to_compare) ** 2))  # /(image_size**2*256)
                if (dist < dist_min):
                    dist_min = dist
                    x_min = i
                    y_min = j

            diff_mat[i_diff][j_diff] = -dist
            j_diff += 1
        j_diff = 0
        i_diff += 1


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
            if(len(x_list)==1):
                x_min = i
                y_min = j
                return x_min, y_min, diff_mat
            image_to_compare = mid_image[i:i + image_len, j:j + image_width]
            dist = (np.sum((image - image_to_compare) ** 2)) / (image_len * image_width * 256)
            if (dist < dist_min):
                dist_min = dist
                x_min = i
                y_min = j

    #if (not same_year):
    #    x_min = x_min + 35
    #    y_min = y_min + 50

    return x_min, y_min, diff_mat

def generate_large_image(road_alg=0,ulman_alg=0,same_year=0):
    if (road_alg):
        image_2020 = cv.imread('road_2020.png')          # queryImage
        image_2018 = cv.imread('road_2018.png')          # queryImage
    elif(ulman_alg):
        image_2020 = cv.imread('ulman_mid_im.png')          # queryImage
        image_2018 = cv.imread('ulman_mid_im.png')          # queryImage
    else:
        raster_path_2020 = "wetransfer-e55797 2020/ecw1.tif"
        if (same_year):
            raster_path_2018 = "wetransfer-e55797 2020/ecw1.tif"
        else:
            raster_path_2018 = "wetransfer-8334c6 2018/TechnionOrtho20181.tif"
        im = tifffile.imread(raster_path_2020)[:][:][1:]
        image_2020 = np.array(im)
        im_18 = tifffile.imread(raster_path_2018)[:][:][1:]
        image_2018 = np.array(im_18)
    return image_2018,image_2020

def generate_mid_image(image_2018_large,image_2020_large,mid_image_len,mid_image_width,road_alg=0,blur_image=1,thresh_alg=0,ulman_alg=0):

        #plt.figure(5)
        #plt.imshow(im_18)
        #plt.show()
        #plt.figure(6)
        #plt.imshow(im)
        #plt.show()
        if(road_alg | ulman_alg):
            image_2020 = image_2020_large
            image_2018 = image_2018_large
        else:
            images = sim_func.create_images(image_2020_large,image_2018_large,mid_image_len,mid_image_width)
            image_2020 = images[0]
            image_2018 = images[2]

        if(blur_image):
            kernel = np.ones((10, 10), np.float32) / 100
            image_2020 = cv.filter2D(image_2020, -1, kernel)
            image_2018 = cv.filter2D(image_2018, -1, kernel)
        if (thresh_alg):
            median_2018 = np.median(image_2018)
            median_2020 = np.median(image_2020)

        return image_2020,image_2018
    #if (hog_alg):
    #    fd, hog_road_2020 = hog(road_2020, orientations=12, pixels_per_cell=(8, 8),
    #                        cells_per_block=(8, 8), visualize=True, multichannel=True)
    #    road_2020 = np.array(hog_road_2020)
    #    fd, hog_road_2018 = hog(road_2018, orientations=12, pixels_per_cell=(8, 8),
    #                        cells_per_block=(8, 8), visualize=True, multichannel=True)
    #    road_2018 = np.array(hog_road_2018)
    #elif (grey_alg):
    #    road_2020 = cv.cvtColor(road_2020, cv.COLOR_BGR2GRAY)
    #    road_2018 = cv.cvtColor(road_2018, cv.COLOR_BGR2GRAY)
    #    if (thresh_alg):
    #        road_2020[road_2020>127] = 255
    #        road_2020[road_2020<=127] = 0
    #        road_2018[road_2018>127] = 255
    #        road_2018[road_2018<=127] = 0


def generate_uvm_image(mid_image,image_len,image_width,rotate_im=1,road_alg=0,ulman_alg=0):

    #creat UVM image:
     if (road_alg):
        y_orig = random.randint(500,1400)
        x_orig = random.randint(400,500)
     else:
        y_orig = random.randint(0,len(mid_image)-image_len)
        x_orig = random.randint(0,len(mid_image[0])-image_width)


     image = mid_image[x_orig:x_orig + image_len, y_orig:y_orig + image_width]
     if(ulman_alg):
         image = cv.imread('ulman_small_im_1.jpeg')  # queryImage
         #image = cv.resize(image, dsize=(image_len, image_width), interpolation=cv.INTER_CUBIC)
     if (rotate_im):
        ang = random.randint(10,60)
        print("angle of rotate is:"+str(ang))
        image = rotate_image.rotate(image,ang)
     return image,x_orig,y_orig

def generate_im_to_show(mid_im,x_cor,y_cor,image_len,image_witdh):
    mid_im[x_cor:x_cor + 5, y_cor:y_cor + image_witdh, 0] = 255
    mid_im[x_cor + image_len:x_cor + image_len + 5, y_cor:y_cor + image_witdh, 0] = 255
    mid_im[x_cor:x_cor + image_len, y_cor:y_cor + 5, 0] = 255
    mid_im[x_cor:x_cor + image_len, y_cor + image_witdh:y_cor + image_witdh + 5, 0] = 255
    return mid_im





def estimate_curr_uvm_cor(uvm_image_cor,mid_image_cor,large_image):
    # configurations:
    step_size = 12
    mid_image_size = 600
    uvm_image_size = 150
    mid_image = large_image[mid_image_cor[0]-int(mid_image_size/2) : mid_image_cor[0]+int(mid_image_size/2) , mid_image_cor[1]-int(mid_image_size/2) : mid_image_cor[1]+int(mid_image_size/2)]
    uvm_image = large_image[uvm_image_cor[0]-int(uvm_image_size/2) : uvm_image_cor[0]+int(uvm_image_size/2) , uvm_image_cor[1]-int(uvm_image_size/2) : uvm_image_cor[1]+int(uvm_image_size/2)]
    curr_x, curr_y, diff_mat = calc_im_diff(uvm_image, mid_image, step_size, hist_alg=0, hist_col=1,hist_list_len=1, max_color_alg=0,max_span=0, hog_alg=0, mean_hist=0)
    curr_y = curr_y+ int(mid_image_size/2) + int(uvm_image_size/2)
    estimated_curr_uvm_cor = [curr_x,curr_y]
    return estimated_curr_uvm_cor

