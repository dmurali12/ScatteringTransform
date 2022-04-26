import network_function as NF
import visualize_parallel as vispa
import torch
import torch.nn as nn
import torch.nn.functional as F


def run_parallel(feature, direct, L, J, class_names, s2, n_cores, im_size):
    data_dir = direct.format(feature)
    l_array = []
    for orientation_index in range(L):
        output = NF.network_run(data_dir, J, L, class_names, s2, n_cores, im_size, orientation_index)
        parallel = [output[0], output[1]]
        unit_average = parallel[0].mean()
        l_array.append(unit_average)

    return l_array

def run_perpendicular(feature, direct, L, J, class_names, s2, n_cores, im_size):
    data_dir = direct.format(feature)
    l_array = []
    for orientation_index in range(L):
        output = NF.network_run(data_dir, J, L, class_names, s2, n_cores, im_size, orientation_index)
        per = [output[2], output[3]]
        unit_average = per[0].mean()
        l_array.append(unit_average)

    return l_array

def run_quart(feature, direct, L, J, class_names, s2, n_cores, im_size):
    data_dir = direct.format(feature)
    l_array = []
    for orientation_index in range(L):
        output = NF.network_run(data_dir, J, L, class_names, s2, n_cores, im_size, orientation_index)
        per = [output[4], output[5]]
        unit_average = per[0].mean()
        l_array.append(unit_average)

    return l_array