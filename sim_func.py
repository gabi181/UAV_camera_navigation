import random
import cv2 as cv
import tifffile
import rotate_image
import numpy as np
import matplotlib.pyplot as plt
import math


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
        image_2020 = cv.imread('project_drone/road_2020.png')          # queryImage
        image_2018 = cv.imread('project_drone/road_2018.png')          # queryImage
        plt.figure(0)
        plt.imshow(image_2020)
        plt.figure(1)
        plt.imshow(image_2018)
        plt.show()
    elif(ulman_alg):
        image_2020 = cv.imread('ulman_mid_im.png')          # queryImage
        image_2018 = cv.imread('ulman_mid_im.png')          # queryImage
    else:
        raster_path_2020 = "project_drone/wetransfer-e55797 2020/ecw1.tif"
        if (same_year):
            raster_path_2018 = "project_drone/wetransfer-e55797 2020/ecw1.tif"
        else:
            raster_path_2018 = "project_drone/wetransfer-8334c6 2018/TechnionOrtho20181.tif"
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

def affinic_from_2020_to_2018(point_2020):
    # calculted by fitting in matlab using:
    # y_vec_2020 = [4524,5809,6149,7634,4409,7305,7513,2412,12043,14260,5315];
    # x_vec_2020 = [4379,5584,6707,7253,7498,4766,2471,7318,14586,12409,8268];
    # y_vec_2018 = [4471,5782,6145,7642,4423,7262,7421,2422,12203,14373,5344];
    # x_vec_2018 = [4304,5481,6596,7111,7424,4631,2332,7285,14350,12127,8174];
    point_2018 = [0,0]
    point_2018[1] = int(-0.02099*point_2020[0] + 0.9998*point_2020[1] + 19.67)  # axis y in np array - x in showing image
    point_2018[0] = int(0.9998*point_2020[0] + 0.02085*point_2020[1] + -142.1)  # axis x in np array - y in showing image

    # for debug:
    # raster_path_2020 = "wetransfer-e55797 2020/ecw1.tif"
    # raster_path_2018 = "wetransfer-8334c6 2018/TechnionOrtho20181.tif"
    # im = tifffile.imread(raster_path_2020)[:][:][1:]
    # image_2020 = np.array(im)
    # im_18 = tifffile.imread(raster_path_2018)[:][:][1:]
    # image_2018 = np.array(im_18)
    # point_2020 = [10200, 8200]
    # point_2018 = convert_2020_to_2018(point_2020)
    # plt.figure(0)
    # plt.imshow(image_2020[point_2020[0]:point_2020[0] + 200, point_2020[1]:point_2020[1] + 200, :])
    # plt.figure(1)
    # plt.imshow(image_2018[point_2018[0]:point_2018[0] + 200, point_2018[1]:point_2018[1] + 200, :])
    # plt.show()
    return point_2018


def small_rand_rotate(im, seed=1):
    random.seed(seed)
    angle = random.randint(0, 5)
    im = rotate_image.rotate(im, angle)
    return im


def generate_true_points(initial_p, im_shape, uav_im_shape, N, general_direction, step_ratio):
    np.random.seed(initial_p[0] + initial_p[1])
    max_step = np.round(np.array(uav_im_shape) / step_ratio).astype(int)
    pts = np.zeros((N,2), int)
    pts[0] = initial_p
    for i in range(1, N):
        if general_direction == 'RD':
            row = pts[i-1, 0] + np.random.randint(0, max_step[0])
            col = pts[i-1, 1] + np.random.randint(0, max_step[1])
        elif general_direction == 'R':
            row = pts[i-1, 0] + np.random.randint(-int(max_step[0]/5), int(max_step[0]/5))
            col = pts[i-1, 1] + np.random.randint(0, max_step[1])
        elif general_direction == 'RU':
            row = pts[i - 1, 0] - np.random.randint(0, max_step[0])
            col = pts[i - 1, 1] + np.random.randint(0, max_step[1])
        elif general_direction == 'U':
            row = pts[i - 1, 0] - np.random.randint(0, max_step[0])
            col = pts[i-1, 1] + np.random.randint(-int(max_step[1]/5), int(max_step[1]/5))
        elif general_direction == 'LU':
            row = pts[i - 1, 0] - np.random.randint(0, max_step[0])
            col = pts[i - 1, 1] - np.random.randint(0, max_step[1])
        elif general_direction == 'L':
            row = pts[i - 1, 0] + np.random.randint(-int(max_step[0] / 5), int(max_step[0] / 5))
            col = pts[i - 1, 1] - np.random.randint(0, max_step[1])
        elif general_direction == 'LD':
            row = pts[i - 1, 0] + np.random.randint(0, max_step[0])
            col = pts[i - 1, 1] - np.random.randint(0, max_step[1])
        elif general_direction == 'D':
            row = pts[i - 1, 0] + np.random.randint(0, max_step[0])
            col = pts[i - 1, 1] + np.random.randint(-int(max_step[1] / 5), int(max_step[1] / 5))
        # The coordinates should't exceed the image bounds.
        row = row * (row < (im_shape[0] - uav_im_shape[0] / 2)) + pts[i - 1, 0] * (
                    row > (im_shape[0] - uav_im_shape[0] / 2))
        col = col * (col < (im_shape[1] - uav_im_shape[1] / 2)) + pts[i - 1, 1] * (
                    col > (im_shape[1] - uav_im_shape[1] / 2))
        pts[i] = np.array([row, col])
    return pts


def getPoints(im, N):
    plt.figure()
    plt.imshow(im)
    pts = np.round(np.array(plt.ginput(N, timeout=120))).astype(int)
    pts = np.flip(pts, axis=1)
    plt.close()
    return pts


def calc_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)  # Pythagorean theorem


def transform_pts(p, H):
    p_hetro = np.expand_dims(np.insert(np.flip(p), 2, 1), axis=1)
    p_prime = np.squeeze(np.round(np.flip((H @ p_hetro).T)).astype(int))
    return p_prime
