import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.feature import hog

def upperleft2center(upperleft_cor, im_shape):
    """
    :param upperleft_cor: Expects to get row-column coordinates. not x-y.
    :param im_shape: im.shape
    :return: center row-column coordinates.
    """
    center_cor = np.array([round(upperleft_cor[0] + im_shape[0]/2), round(upperleft_cor[1] + im_shape[1]/2)])
    return center_cor


def center2upperleft(center_cor, im_shape):
    """
    :param upperleft_cor: Expects to get row-column coordinates. not x-y.
    :param im_shape: im.shape
    :return: upper-left row-column coordinates.
    """
    upperleft_cor = np.array([round(center_cor[0] - im_shape[0]/2), round(center_cor[1] - im_shape[1]/2)])
    return upperleft_cor

def center2im(center_cor, image, im_shape):
    """
    :param center_cor: Row-column center coordinates.
    :param im_shape:
    :return:
    """
    im = image[center_cor[0]-int(im_shape[0]/2): center_cor[0]+int(im_shape[0]/2),
               center_cor[1]-int(im_shape[1]/2): center_cor[1]+int(im_shape[1]/2)]
    return im


def calc_im_diff(image, mid_image, step_size=17, hist_alg=1, hist_col=40, hist_list_len=1, mean_hist=1,max_color_alg=0,max_span=5,hog_alg=0):
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

<<<<<<< HEAD:func_for_pf_alg.py
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
=======
>>>>>>> 23d409d81bdc7982f425e70de99fda8ccd9b97b9:algorithm_functions.py


def match_with_sift(med_im,small_im):
    """
    :param med_im: Image of the searching area.
    :param small_im: Image observed by the uav.
    :return: uav_cor: Coordinates of the UAV in mid image coordinates system.
             H: Affine transformation from small_im to mid_im. x-y coordinates.
    """
    med_im = cv.normalize(med_im, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    small_im = cv.normalize(small_im, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(small_im,None)
    kp2, des2 = sift.detectAndCompute(med_im,None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    """
    For debug:
    cv.drawMatchesKnn expects list of lists as matches.
    good_list = [[i] for i in good]
    img3 = cv.drawMatchesKnn(small_im,kp1,med_im,kp2,good_list,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure()
    plt.imshow(img3)
    """
    if len(good) == 0:
        return [False, False]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inliers = cv.estimateAffine2D(src_pts, dst_pts, method=cv.RANSAC)
    # TODO: I think we should get only translation. So maybe we should check if the 2x2 matrix is close to identity.
    #       im not sure this assumption is true. maybe only affine transformation can explain the differeneces.
    # np.linalg.norm(H[:, [0, 1]] - np.identity(2))
    # TODO: check if u need affine transormation, and if u do - use it to return the uav coordinate it.
    p = np.concatenate((np.flip(np.round(np.array([small_im.shape[0:2]])/2).astype(int)).T, np.ones((1, 1), int)))
    uav_cor = np.round(np.flip((H @ p).T)).astype(int)
    """
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(med_im)
    ax[1].imshow(small_im)
    p = np.array([[858], [134]])
    A = H[:,[0,1]]
    b = H[:,[2]]
    p_prime = np.round(A @ p + b).astype(int)
    ax[1].scatter(p[0], p[1], marker=".", color="red", s=50)
    ax[0].scatter(p_prime[0], p_prime[1], marker=".", color="red", s=50) 
    """
    return uav_cor, H


def calc_uav_cor(uav_image, prev_cor, large_image):
    """
    This is the user function. Wraps the main algorithm.
    :param uav_image: The input from the UAV camera.
    :param prev_cor:  Previous coordinates to reduce the search area.
    :param large_image: The data base. TODO: maybe we should consider reading the data base in the function.
                                             And think about how to handel the data base (keeping all of it in memory
                                             might be impossible)
    :return: Estimated current coordinates.
    """
    mid_image_shape = uav_image.shape[0]*3, uav_image.shape[1]*3
    # TODO: extract the mid image from data base
    mid_image = center2im(prev_cor, large_image, mid_image_shape)
    est_mid_cor, _ = match_with_sift(mid_image, uav_image)
    upperleft_prev_cor = center2upperleft(prev_cor, mid_image_shape)
    est_large_cor = upperleft_prev_cor + est_mid_cor
    return est_large_cor

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
