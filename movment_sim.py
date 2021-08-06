from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import algorithm_functions
#%% parameters
uav_image_size = (80, 80)
#%%
p = Path('.')
q = p.resolve().parent / 'data' / 'simple_data'

im_num = sum([True for f in q.iterdir() if f.is_file()])

images = []
for i,f in enumerate(q.iterdir()):
    if f.is_file():
        images.append(plt.imread(f))


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
true_points = np.array([[436,  886], [444, 1018], [447, 1114], [447, 1215], [436, 1320]])

#%%
# Use the Algorithm to calculate estimated Drone location.
#est_pts = []
est_pts = np.zeros(true_points.shape).astype(int)
est_pts[0] = true_points[0]  # First location is given
for i in range(1,len(true_points)):
    uav_image = algorithm_functions.center2im(true_points[i], images[0], uav_image_size)
    est_pts[i] = algorithm_functions.calc_uav_cor(uav_image=uav_image, prev_cor=est_pts[i-1], large_image=images[0])


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
