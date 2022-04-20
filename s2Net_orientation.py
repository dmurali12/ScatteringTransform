import torch
import torch.nn as nn

# The coordinate (lx, jx, ly, jy) refers to (orientation of corr. first layer channel, scale of corr. first layer
# channel, orientation of second layer channel, scale of channel). Given lx, find all parallel ly)

class s2Net(nn.Module):
    def __init__(self, J, L, layer1_orientation):
        super(s2Net, self).__init__()

        self.l1 = layer1_orientation

        self.norm_index = []

        # Define index for parallel and perpendicular filters

        self.per_layer1_index = []
        self.per_layer2_index = []
        self.par_layer1_index = []
        self.par_layer2_index = []
        self.quarter_layer1_index = []
        self.quarter_layer2_index = []

        for j1 in range(J):
            for j2 in range(j1 + 1, J):
                for l2 in range(L):

                    coeff_index_layer2 = layer1_orientation * L * (J - j1 - 1) + l2 + L * (j2 - j1 - 1) + (L ** 2) * \
                                         (j1 * (J - 1) - j1 * (j1 - 1) // 2)

                    coeff_index_layer1 = layer1_orientation + j1 * L

                    # parallel
                    if layer1_orientation == l2:
                        self.par_layer1_index.append(coeff_index_layer1)
                        self.par_layer2_index.append(coeff_index_layer2)
                    # perpendicular
                    if l2 == (layer1_orientation + (L / 2)) or l2 == (layer1_orientation - (L / 2)):
                        self.per_layer1_index.append(coeff_index_layer1)
                        self.per_layer2_index.append(coeff_index_layer2)
                    # 45 degrees
                    if l2 == (layer1_orientation + (L // 4)) or l2 == (layer1_orientation - (L // 4)):
                        self.quarter_layer1_index.append(coeff_index_layer1)
                        self.quarter_layer2_index.append(coeff_index_layer2)

    def forward(self, x):
        x = x.squeeze()
        sz = x.shape

        # Average pooling across different spatial location
        sti = x.mean((len(sz) - 2, len(sz) - 1))  # 1d array of values

        scat_coeffs_parallel_order_2 = sti[self.par_layer2_index,] / sti[self.par_layer1_index,]
        scat_coeffs_per_order_2 = sti[self.per_layer2_index,] / sti[self.per_layer1_index,]
        scat_coeffs_quart_order_2 = sti[self.quarter_layer2_index,] / sti[self.quarter_layer1_index,]

        # Return normalized s2 coefficient - activation
        return [self.par_layer2_index, scat_coeffs_parallel_order_2], [self.per_layer2_index, scat_coeffs_per_order_2], [self.quarter_layer2_index, scat_coeffs_quart_order_2]
