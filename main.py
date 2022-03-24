import ScatteringTransCurr as ST
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms
from kymatio.torch import Scattering2D
import torchvision.transforms.functional as functional
import numpy as np
from kymatio.torch import Scattering2D
from PIL import Image
import os, re, sys
import umap

import matplotlib.pyplot as plt

class_names = ['IM', ]  # class of image ('apple, orange, etc')
s2 = {'name': [], 'par': [], 'per': [], 'quart': []}

n_cores = 0
im_size = (256, 256)
J = 5  # number of scales
L = 8  # number of orientations

dirs_I = ['45']
j_I = {}

# j_array for I
for ele in dirs_I:
    data_dir = '/Users/dixshetamuralikrishnan/Documents/McCloskey SP22/ConcaveSimple/I_{}'.format(ele)
    j_array = []
    for j1 in range(J):
        data = ST.s2Net_j(J, L, j1)
        parallel = ST.network_parallel(data_dir, J, L, class_names, s2, n_cores, im_size)
        j1_average = parallel.mean()
        j_array.append(j1_average)
    j_I[j1] = j_array

print(j_I)

