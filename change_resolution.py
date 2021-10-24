import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature


def change_resolution(image, resize_value):
    resize_len = resize_value
    resize_width = resize_value
    image_len = len(image[0, :])
    image_width = len(image[:, 0])
    size = int(image_len/resize_len), int(image_width/resize_width)
    image_resized = cv.resize(image, size)
    return image_resized

def add_noise(image):
    # Generate noisy image of a square
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = ndi.rotate(image, 0, mode='constant')
    image = ndi.gaussian_filter(image, 4)
    image = random_noise(image, mode='speckle', mean=0.1)
    return image

