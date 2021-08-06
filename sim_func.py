import random
import cv2 as cv
import tifffile
import random
import rotate_image
import numpy as np


def create_images(im,im_18,med_im_width = 1000, med_im_height = 1000):
    small_im_width = 120
    small_im_height = 120
    #x_cor_med = random.randint(0,im.shape[0]-med_im_width)
    #y_cor_med = random.randint(0,im.shape[1]-med_im_height)
    x_cor_med = random.randint(9000,9500)
    y_cor_med = random.randint(6000,6500)

    x_cor_small = random.randint(x_cor_med,x_cor_med+med_im_width)
    y_cor_small = random.randint(y_cor_med,y_cor_med+med_im_height)

    med_im = im[x_cor_med:x_cor_med+med_im_width,y_cor_med:y_cor_med+med_im_height,:]
    small_im = im[x_cor_small:x_cor_small+small_im_width,y_cor_small:y_cor_small+small_im_height,:]
    x_cor_med = x_cor_med
    y_cor_med = y_cor_med
    med_im_18 = im_18[x_cor_med:x_cor_med + med_im_width, y_cor_med:y_cor_med + med_im_height, :]
    small_im_18 = im_18[x_cor_small:x_cor_small + small_im_width, y_cor_small:y_cor_small + small_im_height, :]
    med_im_cor = [x_cor_med, y_cor_med]
    small_im_cor = [x_cor_small, y_cor_small]
    return [med_im, small_im, med_im_18, small_im_18, med_im_cor, small_im_cor]


def return_road_image(im):
    # these specific coordinates and size fits to Newe Shaanan gate - Kikar Havectorim.
    med_im_width = 4000
    med_im_height = 1000
    x_cor_med = 2000
    y_cor_med = 4200
    road_im = im[y_cor_med:y_cor_med + med_im_height, x_cor_med:x_cor_med + med_im_width, :]
    return road_im


    med_im_cor = [x_cor_med,y_cor_med]
    small_im_cor = [x_cor_small,y_cor_small]
    return [med_im,small_im,med_im_cor,small_im_cor]


def return_road_image(im):
    # these specific coordinates and size fits to Newe Shaanan gate - Kikar Havectorim.
    med_im_width = 4000
    med_im_height = 1000
    x_cor_med = 2000
    y_cor_med = 4200
    road_im = im[y_cor_med:y_cor_med + med_im_height, x_cor_med:x_cor_med + med_im_width, :]
    return road_im

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
            images = create_images(image_2020_large,image_2018_large,mid_image_len,mid_image_width)
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