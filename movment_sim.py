from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytz
from datetime import datetime
from matplotlib.lines import Line2D

import algorithm_functions
import change_resolution
import sim_func
import uav_control

# %% parameters

resize_ratio = 2  # TODO: maybe resize ratio should be algorithm parameter not simulation parameter.
                  #     if so, we should think how to avoid computing the decimation each time.
searching_area_ratio = 2
height = (int(400 / resize_ratio), int(400 / resize_ratio))
rotate = False
step_ratio = 4
max_step = np.round(np.array(height) / step_ratio).astype(int)
save = False
velocity = 20  # pixels per sec
frame_rate = 1  # 1 / sec
dest_thresh = 10
wind_direction = np.pi
wind_max_strength = 5
noise = 0.5

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
N = 3
# uav_path = sim_func.getPoints(images[0], N)
uav_path = np.array([[597, 390], [444, 390], [335, 323]])
# first_coo = sim_func.getPoints(images[0], 1).squeeze()
first_coo = np.array([477, 478])
# true_points = sim_func.generate_true_points(first_coo, images[0].shape, uav_image_size, N, 'RD', step_ratio)
# true_points = np.array([[2401, 1237],[2395, 1253],[2395, 1270],[2395, 1286],[2390, 1292]]) # for east_image

#############################
# %% fix differences between different years databases.
#############################

hetro_true_points = np.concatenate((np.flip(uav_path).T, np.ones((1, len(uav_path)), int)))  # [x, y, 1].T
# H: right -> left
_, H = algorithm_functions.match_with_sift(images[1], images[0])
# H = np.array([[0.9998136, 0.02087663, 8.69967255], [-0.0208969, 0.99982217, 4.59428211]])  # H for east_image
true_points_prime = np.round(np.flip((H @ hetro_true_points).T)).astype(int)  # [R, C]
"""
fig, ax = plt.subplots(2, 1)
ax[0].imshow(images[0])
ax[1].imshow(images[1])
ax[0].scatter(true_points[:,1], true_points[:,0], marker=".", color="red", s=50)
ax[1].scatter(true_points_prime[:,1], true_points_prime[:,0], marker=".", color="red", s=50)
"""
############################################################
# %% Objects creation.
############################################################
search_params = uav_control.Search_Params(height, resize_ratio, searching_area_ratio)
shifts = uav_control.Shift(wind_direction, wind_max_strength, noise)
uav = uav_control.Uav(velocity, frame_rate, uav_path[0], images[0], search_params, dest_thresh)
############################################################
# %% Use the Algorithm to calculate estimated Drone location.
############################################################
est_locations = [uav_path[0]]
true_locations = [uav_path[0]]
for dest in uav_path[1:]:
    uav.set_dest(dest)
    while not uav.arrived:
        direction = uav.calc_direction()
        new_location = uav.move(direction, shifts, true_locations[-1])
        true_locations.append(new_location)
        # Transform database coordinates to environment coordinates system. simulative only
        new_location_hetro = np.expand_dims(np.insert(np.flip(new_location), 2, 1), axis=1)
        new_location_prime = np.squeeze(np.round(np.flip((H @ new_location_hetro).T)).astype(int))
        uav_image = algorithm_functions.center2im(new_location_prime, images[1], uav.search_params.height)
        if rotate:
            uav_image = sim_func.small_rand_rotate(uav_image)
        est_locations.append(uav.update_location(uav_image))

######################
# %% Animation Plot
######################
est_locations = np.array(est_locations)
true_locations = np.array(true_locations)
fig1 = plt.figure(5)
plt.subplot(1,2,1)
line1 = Line2D([0], [0], color='blue', linewidth=3, linestyle='dotted')
line2 = Line2D([0], [0], color='red', linewidth=1, linestyle='-')
labels = ['Estimated Points', 'Searching Area']
plt.legend([line1, line2], labels)
plt.title("Data Base Image")
plt.imshow(images[0])
plt.subplot(1,2,2)
line3 = Line2D([0], [0], color='blue', linewidth=3, linestyle='dotted')
line4 = Line2D([0], [0], color='green', linewidth=1, linestyle='-')
labels = ['Real Points', 'UAV Image']
plt.legend([line3, line4], labels)
plt.title("Real World")
plt.imshow(images[1])
plt.subplot(1,2,2)
for i in range(len(true_locations)):
    if(i != 0):
        rectangle.remove()
    col= max(est_locations[i][1]-height[1]/2,0)
    col_size = est_locations[i][1] - col + min(height[1]/2,images[0].shape[1] - est_locations[i][1])
    row= max(est_locations[i][0]-height[0]/2,0)
    row_size = est_locations[i][0] - row + min(height[0]/2,images[0].shape[0] - est_locations[i][0])
    rectangle = plt.Rectangle((col, row), col_size, row_size, fill=None,ec="red")
    plt.subplot(1,2,1)
    plt.gca().add_patch(rectangle)
    plt.scatter(est_locations[:i+1, 1], est_locations[:i+1, 0], marker=".", color="blue", s=50)
    plt.subplot(1,2,2)
    plt.scatter(true_locations[:, 1], true_locations[:, 0], marker=".", color="red", s=50)
    plt.pause(0.01)



############################
# %% Statistics and results.
############################
# plot true points and estimated points.
fig1 = plt.figure(1)
plt.imshow(images[0])
plt.scatter(true_locations[:, 1], true_locations[:, 0], marker=".", color="red", s=50)
plt.scatter(est_locations[:, 1], est_locations[:, 0], marker=".", color="blue", s=50)

dists = []
for i in range(len(est_locations)):
    dists.append(sim_func.calc_distance(est_locations[i], true_locations[i]))
avg_err = int(sum(dists)/len(dists))
# plot distances
fig2 = plt.figure(2)
plt.plot(dists, 'bo')
text = ("Avg Err = " + str(avg_err) + '\t' +
        "UAV picture size = " + str(height) + '\t' +
        "Search area ratio = " + str(searching_area_ratio) + '\t' +
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
