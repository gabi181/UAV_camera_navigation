from scipy import ndimage
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math
import imutils
#img1 = cv.imread('Figure_2_rotated.png',cv.IMREAD_GRAYSCALE)

def rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """
    angle = math.radians(angle)
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (bb_w - 2 * x, bb_h - 2 * y)

def crop(img, w, h):
    x, y = int(img.shape[1] * .5), int(img.shape[0] * .5)

    return img[
        int(np.ceil(y - h * .5)) : int(np.floor(y + h * .5)),
        int(np.ceil(x - w * .5)) : int(np.floor(x + h * .5))
    ]

def rotate(img, angle):
    # rotate, crop and return original size
    (h, w) = img.shape[:2]
    img = imutils.rotate_bound(img, angle)
    img = crop(img, *rotated_rect(w, h, angle))
    img = cv.resize(img,(w,h),interpolation=cv.INTER_AREA)
    return img
    #rotation angle in degree
#rotated = rotate(img1, 160)


#plt.imshow(rotated),plt.show()

