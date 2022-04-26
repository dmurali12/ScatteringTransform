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


def setup(im_size, class_names, data_dir, n_cores, J, L):
    # transform the data
    data_transforms = {x: transforms.Compose([
        transforms.Resize(im_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]) for x in class_names
    }

    image_datasets = {x: IF.ImageFolderWithPaths(data_dir,
                                                 data_transforms[x]) for x in class_names
                      }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                                  shuffle=False, num_workers=n_cores) for x in class_names
                   }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_sizes = {x: len(image_datasets[x]) for x in class_names}

    # Define scattering transform
    scattering = Scattering2D(J=J, shape=im_size, L=L, max_order=2)

    return dataloaders, device, dataset_sizes, scattering
