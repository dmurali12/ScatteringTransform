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
import ImageFolderWithPaths
import s2Net_orientation as network

def network_perpendicular(data_dir, J, L, class_names, s2, n_cores, im_size):
    data_transforms = {x: transforms.Compose([
        transforms.Resize(im_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]) for x in class_names
    }

    image_datasets = {x: ImageFolderWithPaths(data_dir,
                                              data_transforms[x]) for x in class_names
                      }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                                  shuffle=False, num_workers=n_cores) for x in class_names
                   }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_sizes = {x: len(image_datasets[x]) for x in class_names}

    # Define scattering transform
    scattering = Scattering2D(J=J, shape=im_size, L=L, max_order=2)

    # Move it to GPU if available
    if torch.cuda.is_available():
        scattering = scattering.cuda()

    for j1 in range(J - 1):
        s2_Net = s2Net_j(j1)
        with torch.no_grad():
            for label in class_names:
                print(label)
                inputs = dataloaders[label]
                for i, data in enumerate(inputs, 0):
                    data, n1, path = data
                    # Get the name of the image from the path
                    label = path[0]
                    label = re.split("/", label)
                    label = label[-1]

                    # Load image into device
                    data = data.to(device)
                    # Run network
                    output_parallel, output_per, output_quart = s2_Net(scattering(data))
                    # output = scattering(data)
                    s2['name'].append(label)
                    s2['per'] = np.append(s2['per'], output_per.numpy())
    s2['per'] = np.asarray(s2['per'])
    return s2['per']