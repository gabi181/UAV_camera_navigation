import random


def create_images(im):
    med_im_width = 1000
    med_im_height = 1000
    small_im_width = 400
    small_im_height = 400
    x_cor_med = random.randint(0,im.shape[0]-med_im_width)
    y_cor_med = random.randint(0,im.shape[1]-med_im_height)

    x_cor_small = random.randint(x_cor_med,x_cor_med+med_im_width)
    y_cor_small = random.randint(y_cor_med,y_cor_med+med_im_height)

    med_im = im[x_cor_med:x_cor_med+med_im_width,y_cor_med:y_cor_med+med_im_height]
    small_im = im[x_cor_small:x_cor_small+small_im_width,y_cor_small:y_cor_small+small_im_height]
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

