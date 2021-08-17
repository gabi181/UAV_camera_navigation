from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytz
from datetime import datetime

import algorithm_functions
import change_resolution
import sim_func

# %% parameters

resize_ratio = 1  # TODO: maybe resize ratio should be algorithm parameter not simulation parameter.
#     if so, we should think how to avoid computing the decimation each time.
mid_ratio = 3
uav_image_size = (int(500 / resize_ratio), int(500 / resize_ratio))
rotate = False
step_ratio = 4
max_step = np.round(np.array(uav_image_size) / step_ratio).astype(int)
save = False

# %% configuration
cfg_rand_rotate = 0
# %%
# TODO: maybe we should consider reading the data base in the function.
#  And think about how to handel the data base (keeping all of it in memory might be impossible)
p = Path('.')
# q = p.resolve().parent / 'data' / 'simple_data'
q = p.resolve().parent / 'data' / 'east_data'
save_path = p.resolve().parent / 'results'
im_num = sum([True for f in q.iterdir() if f.is_file()])

images = []
for f in q.iterdir():
    if f.is_file():
        images.append(plt.imread(f))
        database = f.name
for i, im in enumerate(images):
    images[i] = change_resolution.change_resolution(im, resize_ratio)


#############################
# %% generate path points.
#############################
N = 10
#true_points = getPoints(images[0], N)
# first_coo = sim_func.getPoints(images[0], 1).squeeze()
first_coo = np.array([611, 391])
true_points = sim_func.generate_true_points(first_coo, images[0].shape, uav_image_size, N, 'R', step_ratio)
# true_points = np.array([[152, 878], [153, 887], [153, 899], [150, 907], [147, 918]])  # [R, C]. Resize_ratio = 1.
# true_points = np.array([[85, 390], [85, 382], [85, 374], [85, 367], [85, 359]])  # [R, C]. Resize_ratio = 2.
# true_points = np.array([[2401, 1237],[2395, 1253],[2395, 1270],[2395, 1286],[2390, 1292]]) # for east_image

#############################
# %% fix differences between different years databases.
#############################

hetro_true_points = np.concatenate((np.flip(true_points).T, np.ones((1, len(true_points)), int)))  # [x, y, 1].T
# H: right -> left
# _, H = algorithm_functions.match_with_sift(images[1], images[0])
H = np.array([[0.9998136, 0.02087663, 8.69967255], [-0.0208969, 0.99982217, 4.59428211]])  # H for east_image
true_points_prime = np.round(np.flip((H @ hetro_true_points).T)).astype(int)  # [R, C]
"""
fig, ax = plt.subplots(2, 1)
ax[0].imshow(images[0])
ax[1].imshow(images[1])
ax[0].scatter(true_points[:,1], true_points[:,0], marker=".", color="red", s=50)
ax[1].scatter(true_points_prime[:,1], true_points_prime[:,0], marker=".", color="red", s=50)
"""
############################################################
# %% Use the Algorithm to calculate estimated Drone location.
############################################################
est_pts = np.zeros(true_points.shape).astype(int)
est_pts[0] = true_points[0]  # First location is given
fails_num = 0
for i in range(1, len(true_points)):
    uav_image2020 = algorithm_functions.center2im(true_points_prime[i], images[1], uav_image_size)
    if rotate:
        uav_image2020 = sim_func.small_rand_rotate(uav_image2020)
    # uav_image2018 = algorithm_functions.center2im(true_points[i], images[0], uav_image_size)
    est_pts[i], fails_num = algorithm_functions.calc_uav_cor(uav_image=uav_image2020, prev_cor=est_pts[i - 1],
                                                             large_image=images[0],
                                                             mid_ratio=mid_ratio, fails_num=fails_num)
# %%

######################
# %% Animation Plot
######################
fig1 = plt.figure(5)
plt.imshow(images[0])
plt.scatter(true_points[:, 1], true_points[:, 0], marker=".", color="red", s=50)
for i in range(len(true_points)):
    if(i != 0):
        rectangle.remove()
    col= max(est_pts[i][1]-uav_image_size[1]/2,0)
    col_size = est_pts[i][1] - col + min(uav_image_size[1]/2,images[0].shape[1] - est_pts[i][1])
    row= max(est_pts[i][0]-uav_image_size[0]/2,0)
    row_size = est_pts[i][0] - row + min(uav_image_size[0]/2,images[0].shape[0] - est_pts[i][0])
    rectangle = plt.Rectangle((col, row), col_size, row_size, fill=None,ec="red")
    plt.gca().add_patch(rectangle)
    plt.scatter(est_pts[:i+1, 1], est_pts[:i+1, 0], marker=".", color="blue", s=50)
    plt.pause(0.01)



############################
# %% Statistics and results.
############################
# plot true points and estimated points.
fig1 = plt.figure(1)
plt.imshow(images[0])
plt.scatter(true_points[:, 1], true_points[:, 0], marker=".", color="red", s=50)
plt.scatter(est_pts[:, 1], est_pts[:, 0], marker=".", color="blue", s=50)

dists = []
for i in range(len(est_pts)):
    dists.append(sim_func.calc_distance(est_pts[i], true_points[i]))
avg_err = int(sum(dists)/N)
# plot distances
fig2 = plt.figure(2)
plt.plot(dists, 'bo')
text = ("Avg Err = " + str(avg_err) + '\t' +
        "UAV picture size = " + str(uav_image_size) + '\t' +
        "Search area ratio = " + str(mid_ratio) + '\t' +
        "Rotate = " + str(rotate) + '\t' +
        "reduced resolution by " + str(resize_ratio) + '\t' +
        "step_ratio = " + str(step_ratio) + "\t" +
        "max step size = " + str(int(sim_func.calc_distance(max_step, (0, 0)))))

plt.title(text)


##############################
# Save figures and statistics.
##############################
if save:
    tz = pytz.timezone('Asia/Jerusalem')
    israel_datetime = datetime.now(tz)

    israel_datetime.strftime("%d-%m_%H-%M")

    figures_doc_file = open(save_path / 'figures_doc_file.txt', 'a')
    figures_doc_file.write('\n' + israel_datetime.strftime("%d-%m_%H-%M") + '\n' +
                           database + '\n' +
                           text + '\t' +
                           "Points number = " + str(N) + '\t' +
                           "First Coordinates = " + str(first_coo)
                           )
    figures_doc_file.close()
    plt.figure(1)
    plt.savefig(save_path / ('Path_' + israel_datetime.strftime("%d-%m_%H-%M") + '.jpg'))
    plt.figure(2)
    plt.savefig(save_path / ('Error_' + israel_datetime.strftime("%d-%m_%H-%M") + '.jpg'))

plt.show()
