import numpy as np
import cv2 as cv


def cart2pol(row_col):
    r = np.sqrt(row_col[0]**2 + row_col[1]**2)
    theta = np.arctan2(-row_col[0], row_col[1]) % (2 * np.pi)
    return np.array([r, theta])


def pol2cart(r_theta):
    x = np.round(r_theta[0] * np.cos(r_theta[1])).astype(int)
    y = np.round(r_theta[0] * np.sin(r_theta[1])).astype(int)
    return np.array([-y, x])


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
    upperleft_row = np.maximum(round(center_cor[0] - im_shape[0]/2), 0)
    upperleft_col = np.maximum(round(center_cor[1] - im_shape[1] / 2), 0)
    upperleft_cor = np.array([upperleft_row,upperleft_col])
    return upperleft_cor

def center2im(center_cor, image, im_shape):
    """
    :param center_cor: Row-column center coordinates.
    :param im_shape:
    :param cgf_rand_rotate: if ==1 - rotates image in random angle between 0 to 5
    :return:
    """
    up_row = np.maximum(center_cor[0] - int(im_shape[0]/2), 0)
    down_row = np.minimum(center_cor[0]+int(im_shape[0]/2), image.shape[0])
    left_col = np.maximum(center_cor[1] - int(im_shape[1]/2), 0)
    right_col = np.minimum(center_cor[1] + int(im_shape[1] / 2), image.shape[1])

    im = image[up_row: down_row, left_col: right_col]

    return im




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
    # np.linalg.norm(H[:, [0, 1]] - np.identity(2))
    p = np.concatenate((np.flip(np.round(np.array([small_im.shape[0:2]])/2).astype(int)).T, np.ones((1, 1), int)))
    uav_cor = np.round(np.flip((H @ p).T)).astype(int).squeeze()
    """
    For debug
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(med_im)
    ax[1].imshow(small_im)
    pt = np.array([[100], [100]])
    A = H[:,[0,1]]
    b = H[:,[2]]
    p_prime = np.round(A @ pt + b).astype(int)
    ax[1].scatter(p[0], p[1], marker=".", color="red", s=50)
    ax[0].scatter(p_prime[0], p_prime[1], marker=".", color="red", s=50) 
    """
    return uav_cor, H # [R, C]


def calc_uav_cor(uav_image, prev_cor, large_image, mid_ratio, fails_num):
    """
    This is the user function. Wraps the main algorithm.
    :param uav_image: The input from the UAV camera.
    :param prev_cor:  Previous coordinates to reduce the search area.
    :param large_image: The data base.
    :param mid_ratio: uav image shape is multiplied by mid ratio to create the searching area.
    :return: Estimated current coordinates.
    """
    if fails_num > 2:
        mid_ratio = 2 * mid_ratio
    mid_image = center2im(prev_cor, large_image, np.array(uav_image.shape)*mid_ratio)
    est_mid_cor, _ = match_with_sift(mid_image, uav_image)
    upperleft_prev_cor = center2upperleft(prev_cor, np.array(uav_image.shape)*mid_ratio)
    est_large_cor = upperleft_prev_cor + est_mid_cor
    """
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(mid_image)
    ax[1].imshow(large_image)
    ax[0].scatter(est_mid_cor[1], est_mid_cor[0], marker=".", color="red", s=50)
    ax[1].scatter(est_large_cor[1], est_large_cor[0], marker=".", color="red", s=50)
    """
    # Edge cases:
    if est_mid_cor[0] > mid_image.shape[0] or est_mid_cor[1] > mid_image.shape[1] or\
            est_mid_cor[0] < 0 or est_mid_cor[1] < 0:
        est_large_cor = prev_cor
        fails_num += 1
    else:
        fails_num = 0
    return est_large_cor, fails_num

