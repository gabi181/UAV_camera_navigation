import cv2 as cv
import matplotlib.pyplot as plt
#img1 = cv.imread('Figure_2_rotated.png',cv.IMREAD_GRAYSCALE)          # queryImage
#img2 = cv.imread('Figure_1.png',cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector

def match_with_sift(med_im,small_im):
    med_im = cv.cvtColor(med_im, cv.COLOR_BGR2GRAY)
    med_im = cv.normalize(med_im, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    small_im = cv.cvtColor(small_im, cv.COLOR_BGR2GRAY)
    small_im = cv.normalize(small_im, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(small_im,None)
    kp2, des2 = sift.detectAndCompute(med_im,None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(small_im,kp1,med_im,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(1)
    plt.imshow(img3)

    x = 0
    y = 0
    for m in good:
        x += kp2[m[0].trainIdx].pt[0]
        y += kp2[m[0].trainIdx].pt[1]
    x = int(round( x / len(good) ))
    y = int(round( y / len(good) ))

    matched_cor = [x,y]
    match_im = med_im[x-100:x+100,y-100:y+100]

    plt.figure(2)
    plt.imshow(match_im)
    plt.show()

    return matched_cor


