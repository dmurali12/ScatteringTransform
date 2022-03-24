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

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


# NN function that
# class s2Net(nn.Module):
#     def __init__(self, J, L):
#         super(s2Net, self).__init__()
#         # order 1 coef length
#         self.len_order_1 = J * L
#
#         # order 2 coef length
#         self.len_order_2 = (J * (J - 1) // 2) * (L ** 2)
#
#         # finding for each s2 unit, the corresponding s1 coefficient (used for normalization)
#         self.norm_index = []
#         # Define index for parallel and perpendicular filters
#
#         self.per_index = []
#         self.parallel_index = []
#         self.quart_index = []
#
#         for j1 in range(J - 1):
#             for j2 in range(j1 + 1, J):  # spatial scale in l2
#                 for l1 in range(L):  # orientiation in l1
#                     for l2 in range(L):
#
#                         coeff_index = l1 * L * (J - j1 - 1) + l2 + L * (j2 - j1 - 1) + (L ** 2) * \
#                                       (j1 * (J - 1) - j1 * (j1 - 1) // 2)
#
#                         coeff_index_1 = l1 + j1 * L
#
#                         self.norm_index.append(coeff_index)
#
#                         # parallel
#                         if l1 == l2:
#                             self.parallel_index.append(coeff_index)
#                         # perpendicular
#                         if l2 == (l1 + 4) or l2 == (l1 - 4):
#                             self.per_index.append(coeff_index)
#
#     def forward(self, x):
#         x = x.squeeze()
#         sz = x.shape
#
#         # Average pooling across different spatial location
#         sti = x.mean((len(sz) - 2, len(sz) - 1))  # 1d array of values
#         # print(sti)
#
#         # Extract scattering coef for first and second layer
#         scat_coeffs_order_1 = sti[1:self.len_order_1, ]  # scat coeff for layer 1
#         # 1d array of values that correspond to layer 1 FROM sti
#         scat_coeffs_order_2 = sti[1 + self.len_order_1:, ]  # scat coeff for l2
#         # Normalization of s2 coef. by s1
#         scat_coeffs_order_2 = scat_coeffs_order_2 / scat_coeffs_order_1[self.norm_index]
#         # You can add something here to slice different orientation/scale for different purpose
#
#         # Return normalized s2 coefficient
#         return scat_coeffs_order_2


# NN function that can look at the 2nd layer units corresponding to a specific orientation
class s2Net_j(nn.Module):
    def __init__(self, J, L, j1):
        super(s2Net_j, self).__init__()

        self.j1 = j1

        self.norm_index = []
        # Define index for parallel and perpendicular filters

        self.per_norm_index = []
        self.per_l2 = []
        self.parallel_norm_index = []
        self.parallel_l2 = []
        self.quart_norm_index = []
        self.quart_l2 = []

        for j2 in range(self.j1 + 1, J):  # spatial scale in l2
            for l1 in range(L):
                for l2 in range(L):

                    coeff_index_l2 = l1 * L * (J - j1 - 1) + l2 + L * (j2 - j1 - 1) + (L ** 2) * \
                                     (j1 * (J - 1) - j1 * (j1 - 1) // 2)

                    coeff_index_l1 = l1 + j1 * L

                    # parallel
                    if l1 == l2:
                        self.parallel_norm_index.append(coeff_index_l1)
                        self.parallel_l2.append(coeff_index_l2)
                    # perpendicular
                    if l2 == (l1 + (L / 2)) or l2 == (l1 - (L / 2)):
                        self.per_norm_index.append(coeff_index_l1)
                        self.per_l2.append(coeff_index_l2)
                    # 45 degrees
                    if l2 == (l1 + (L // 4)) or l2 == (l1 - (L // 4)):
                        self.quart_norm_index.append(coeff_index_l1)
                        self.quart_l2.append(coeff_index_l2)

    def forward(self, x):
        x = x.squeeze()
        sz = x.shape

        # Average pooling across different spatial location
        sti = x.mean((len(sz) - 2, len(sz) - 1))  # 1d array of values

        scat_coeffs_parallel_order_2 = sti[self.parallel_l2,] / sti[self.parallel_norm_index,]
        scat_coeffs_per_order_2 = sti[self.per_l2,] / sti[self.per_norm_index,]
        scat_coeffs_quart_order_2 = sti[self.quart_l2,] / sti[self.quart_norm_index,]

        # Return normalized s2 coefficient
        return scat_coeffs_parallel_order_2, scat_coeffs_per_order_2, scat_coeffs_quart_order_2


def network_parallel(data_dir, J, L, class_names, s2, n_cores, im_size):
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
        s2_Net = s2Net_j(J, L, j1)
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
                    #           output = scattering(data)
                    s2['name'].append(label)
                    s2['par'] = np.append(s2['par'], output_parallel.numpy())

    s2['par'] = np.asarray(s2['par'])

    return s2['par']


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
