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
import ImageFolderWithPaths as IF
import s2Net_orientation as network
import data_setup as DS

import matplotlib.pyplot as plt


# edges
# perpendicular : junctions


def network_run(data_dir, J, L, class_names, s2, n_cores, im_size, layer1_orientation):
    dataloaders, device, dataset_sizes, scattering = DS.setup(im_size, class_names, data_dir, n_cores, J, L)

    # Move it to GPU if available
    if torch.cuda.is_available():
        scattering = scattering.cuda()

    s2_Net = network.s2Net(J, L, layer1_orientation)
    with torch.no_grad():
        for label in class_names:
            print(label)
            inputs = dataloaders[label]
            for i, ele in enumerate(inputs, 0):
                data, n1, path = ele
                # Get the name of the image from the path
                label = path[0]
                label = re.split("/", label)
                label = label[-1]

                # Load image into device
                data = data.to(device)

                # Run network
                output_parallel, output_per, output_quart = s2_Net(scattering(data))
                s2['name'].append(label)
                s2['par'][str(output_parallel[0])] = output_parallel[1]
                s2['per'][str(output_per[0])] = output_per[1]
                s2['quart'][str(output_quart[0])] = output_quart[1]

    # s2['par'] = torch.tensor(s2['par'])

    return output_parallel[1].numpy(), output_parallel[0], output_per[1].numpy(), output_per[0], output_quart[
        1].numpy(), output_quart[0]  # [1] is the activation and [0] is the index
