from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
from func_for_pf_alg import estimate_curr_uav_cor
#%% parameters
uav_image_size = (80, 80)
#%%
p = Path('.')
q = p.resolve().parent / 'data' / 'simple_data'

im_num = sum([True for f in q.iterdir() if f.is_file()])

fig, axes = plt.subplots(im_num, 1, figsize=(200,30))
images = []
for i,f in enumerate(q.iterdir()):
    if f.is_file():
        images.append(cv2.imread(str(f)))
        axes[i].imshow(images[i])
        axes[i].set_xticks([])
        axes[i].set_yticks([])


#%%
# stores mouse position in global variables ix(for x coordinate) and iy(for y coordinate)
# on double click inside the image
def select_point(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:  # captures left button double-click
        true_points.append((x, y))
        cv2.circle(tmp_image, center=true_points[-1], radius=3, color=(0, 0, 255), thickness=-1)


#%%
# Let the user draw points.
true_points = []
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# bind select_point function to a window that will capture the mouse click
cv2.setMouseCallback('image', select_point)
tmp_image = copy.copy(images[0])
while True:
    cv2.imshow('image', tmp_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
true_points = [(sub[1], sub[0]) for sub in true_points]
#%%
# Use the Algorithm to calculate estimated Drone location.
cv2.namedWindow('paths', cv2.WINDOW_NORMAL)
est_pts = []
for pts in true_points:
    uav_image = images[0][pts[0]-int(uav_image_size[0]/2) : pts[0]+int(uav_image_size[0]/2) , pts[1]-int(uav_image_size[1]/2) : pts[1]+int(uav_image_size[1]/2)]
    est_pts.append(estimate_curr_uav_cor(uav_image, pts, images[0]))
    cv2.circle(tmp_image, center=est_pts[-1], radius=3, color=(0, 255, 0), thickness=-1)

while True:
    cv2.imshow('paths', tmp_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


#%%
def calc_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)  # Pythagorean theorem


#%% Statistics
dists = []
for i in range(len(est_pts)):
    dists.append(calc_distance(est_pts[i], true_points[i]))
plt.plot(dists, 'bo')
plt.show()

