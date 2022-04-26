# Goal of project: find how the second layer unit that activates to the same orientation as the corresponding first
# layer unit responds to images that are optimized for specific features

import matplotlib.pyplot as plt

import Run_dict as Rd
from visualize_parallel import plotting as plot

from datetime import date

# Go through images in folders ConcaveSimple/I_(45, 90, or 135).

class_names = ['IM', ]  # class of image ('apple, orange, etc')
s2 = {'name': [], 'par': {}, 'per': {}, 'quart': {}}

n_cores = 0
im_size = (256, 256)
J = 5  # number of scales
L = 8  # number of orientations

# List to access the desired folder of features
dirs_I = ['I_45', 'I_90', 'I_135', 'F_180']
direct = 'ConcaveSimple/{}'

# Add to dictionary of arrays - this defines the feature (45, 90, 135), and the associated average activations for
# the second layer unit. For example, in 45: [a, b, c, d, e], a refers to the activation of the second layer unit
# that's selects for the orientation parallel to the first unit in the first layer
parallel_I = {}
perpendicular_I = {}
quart_I = {}

# f = open("Extracted_Data")
# date = str(date.today())

for feature in dirs_I:
    par_array = Rd.run_parallel(feature, direct, L, J, class_names, s2, n_cores, im_size)
    per_array = Rd.run_perpendicular(feature, direct, L, J, class_names, s2, n_cores, im_size)
    quart_array = Rd.run_quart(feature, direct, L, J, class_names, s2, n_cores, im_size)

    parallel_I[feature] = par_array
    perpendicular_I[feature] = per_array
    quart_I[feature] = quart_array

    print(parallel_I, perpendicular_I, quart_I)


