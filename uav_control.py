import numpy as np
import algorithm_functions


class Shift:
    """
    direction: U, RU, R, RD, D, LD, L, LU,
    strength: pixel per sec
    noise: pixels per sec
    """
    def __init__(self, wind_direction, wind_max_strength, noise):
        self.wind_direction = wind_direction
        self.wind_max_strength = wind_max_strength
        self.noise = noise

    def wind_shift(self):
        """
        :return: Shifts caused by wind and noise.
        """
        if self.wind_max_strength == 0:
            wind_vec = np.array([0, 0])
        else:
            step_size = np.random.randint(int(self.wind_max_strength / 2), self.wind_max_strength)
            wind_vec = algorithm_functions.pol2cart(np.array([step_size, self.wind_direction]))
        return wind_vec

    def noise_shift(self, velocity):
        if velocity == 0 or int(self.noise / velocity) == 0:
            row, col = 0, 0
        else:
            row = np.random.randint(0, int(self.noise / velocity))
            col = np.random.randint(0, int(self.noise / velocity))
        return np.array([row, col])


class Search_Params:
    """
    height: image shape.
    """
    def __init__(self, height, resize_ratio, searching_area_ratio):
        self.height = height
        self.resize_ratio = resize_ratio
        self.searching_area_ratio = searching_area_ratio


class Uav:
    """
    velocity: pixels per sec
    frame_rate: 1 / pixels
    """
    def __init__(self, velocity, frame_rate, start_location, data_base, search_params, dest_thresh):
        self.velocity = velocity
        self.frame_rate = frame_rate
        self.est_curr_location = start_location
        self.data_base = data_base
        self.search_params = search_params
        self.dest_thresh = dest_thresh
        self.arrived = False
        self.destination = np.array([-1, -1])
        self.distance = -1
        self.fails_num = 0

    def set_dest(self, destination):
        self.destination = destination

    def update_location(self, uav_image):
        self.est_curr_location, self.fails_num = \
            algorithm_functions.calc_uav_cor(uav_image, self.est_curr_location,
                                             self.data_base,
                                             self.search_params.searching_area_ratio,
                                             self.fails_num)
        self.distance, _ = algorithm_functions.cart2pol(self.destination - self.est_curr_location)
        if self.distance < self.dest_thresh:
            self.arrived = True
        return self.est_curr_location

    def move(self, direction, shift, true_location):
        np.random.seed(self.est_curr_location[0] + self.est_curr_location[1])
        step_size = np.round(self.velocity / self.frame_rate).astype(int)

        move_vec = algorithm_functions.pol2cart(np.array([step_size, direction]))
        new_loc = true_location + move_vec + shift.noise_shift(self.velocity) + shift.wind_shift()


        # The coordinates should't exceed the image bounds.
        row_out = new_loc[0] > (self.data_base.shape[0] - self.search_params.height[0] / 2) or new_loc[0] < self.search_params.height[0] / 2
        col_out = new_loc[1] > (self.data_base.shape[1] - self.search_params.height[1] / 2) or new_loc[1] < self.search_params.height[1] / 2
        new_loc[0] = new_loc[0] * (1-row_out) + true_location[0] * row_out
        new_loc[1] = new_loc[1] * (1-col_out) + true_location[1] * col_out

        return new_loc

    def calc_direction(self):
        _, theta = algorithm_functions.cart2pol(self.destination - self.est_curr_location)
        return theta



