import parallel_function as PAST
import visualize_parallel as vispa
import torch
import torch.nn as nn
import torch.nn.functional as F

def run_parallel(feature, direct, L, J, class_names, s2, n_cores, im_size):
    data_dir = direct.format(feature)
    l_array = []
    for orientation_index in range(L):
        parallel = PAST.network_parallel(data_dir, J, L, class_names, s2, n_cores, im_size, orientation_index)
        unit_average = parallel[0].mean()
        l_array.append(unit_average)

    return l_array