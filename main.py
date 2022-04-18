import parallel_function as PAST
import torch
import torch.nn as nn
import torch.nn.functional as F

class_names = ['IM', ]  # class of image ('apple, orange, etc')
s2 = {'name': [], 'par': [], 'per': [], 'quart': []}

n_cores = 0
im_size = (256, 256)
J = 5  # number of scales
L = 8  # number of orientations

# List for arrays
dirs_I = ['90']

# Add to dictionary of arrays
orientation_I = {}

for ele in dirs_I:
    data_dir = 'ConcaveSimple/I_{}'.format(ele)
    l_array = []
    for orientation_index in range(L):
        parallel = PAST.network_parallel(data_dir, J, L, class_names, s2, n_cores, im_size, orientation_index)
        unit_average = torch.mean(parallel)
        l_array.append(unit_average)
    orientation_I[int(ele)] = l_array
    print(orientation_I)


