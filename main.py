# Goal of project: find how the second layer unit that activates to the same orientation as the corresponding first
# layer unit responds to images that are optimized for specific features

import parallel_function as PAST
import torch
import torch.nn as nn
import torch.nn.functional as F

# Go through images in folders ConcaveSimple/I_(45, 90, or 135).

class_names = ['IM', ]  # class of image ('apple, orange, etc')
s2 = {'name': [], 'par': {}, 'per': [], 'quart': []}

n_cores = 0
im_size = (256, 256)
J = 5  # number of scales
L = 8  # number of orientations

# List to access the desired folder of features
dirs_I = ['45', '90', '135']

# Add to dictionary of arrays - this defines the feature (45, 90, 135), and the associated average activations for
# the second layer unit. For example, in 45: [a, b, c, d, e], a refers to the activation of the second layer unit
# that's selects for the orientation parallel to the first unit in the first layer
orientation_I = {}

for ele in dirs_I:
    data_dir = 'ConcaveSimple/I_{}'.format(ele)
    l_array = []
    for orientation_index in range(L):
        parallel = PAST.network_parallel(data_dir, J, L, class_names, s2, n_cores, im_size, orientation_index)
        unit_average = parallel.mean()
        l_array.append(unit_average)
    orientation_I[int(ele)] = l_array
    print(orientation_I)


