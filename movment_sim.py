from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import algorithm_functions
import change_resolution

#%% parameters

resize_ratio = 1
uav_image_size = (int(200/resize_ratio), int(200/resize_ratio))

#%%
p = Path('.')
q = p.resolve().parent / 'data' / 'simple_data'

im_num = sum([True for f in q.iterdir() if f.is_file()])

images = []
for f in q.iterdir():
    if f.is_file():
        images.append(plt.imread(f))
for i, im in enumerate(images):
    images[i] = change_resolution.change_resolution(im, resize_ratio)


#%%
def getPoints(im, N):
    plt.figure()
    plt.imshow(im)
    pts = np.round(np.array(plt.ginput(N, timeout=120))).astype(int)
    pts = np.flip(pts, axis=1)
    plt.close()
    return pts


#%%
# Let the user draw points.
N = 5
# true_points = getPoints(images[0], N)
true_points = np.array([[152, 878], [153, 887], [153, 899], [150, 907], [147, 918]])  # [R, C]. Resize_ratio = 1.
# true_points = np.array([[85, 390], [85, 382], [85, 374], [85, 367], [85, 359]])  # [R, C]. Resize_ratio = 2.

hetro_true_points = np.concatenate((np.flip(true_points).T, np.ones((1, len(true_points)), int)))  # [x, y, 1].T
# H: right -> left
_, H = algorithm_functions.match_with_sift(images[1], images[0])
true_points_prime = np.round(np.flip((H @ hetro_true_points).T)).astype(int)  # [R, C]
"""
fig, ax = plt.subplots(2, 1)
ax[0].imshow(images[0])
ax[1].imshow(images[1])
ax[0].scatter(true_points[:,1], true_points[:,0], marker=".", color="red", s=50)
ax[1].scatter(true_points_prime[:,1], true_points_prime[:,0], marker=".", color="red", s=50)
"""

#%%
# Use the Algorithm to calculate estimated Drone location.
est_pts = np.zeros(true_points.shape).astype(int)
est_pts[0] = true_points[0]  # First location is given
for i in range(1,len(true_points)):
    # TODO: do not extract uav image from database.
    uav_image2020 = algorithm_functions.center2im(true_points_prime[i], images[1], uav_image_size)
    # uav_image2018 = algorithm_functions.center2im(true_points[i], images[0], uav_image_size)
    est_pts[i] = algorithm_functions.calc_uav_cor(uav_image=uav_image2020, prev_cor=est_pts[i-1], large_image=images[0])


#%%
def calc_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)  # Pythagorean theorem


#%% Statistics and results
# plot true points and estimated points.
plt.figure(1)
plt.imshow(images[0])
plt.scatter(true_points[:, 1], true_points[:, 0], marker=".", color="red", s=50)
plt.scatter(est_pts[:, 1], est_pts[:, 0], marker=".", color="blue", s=50)

dists = []
for i in range(len(est_pts)):
    dists.append(calc_distance(est_pts[i], true_points[i]))
# plot distances
plt.figure(2)
plt.plot(dists, 'bo')
plt.show()
