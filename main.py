# Goal of project: find how the second layer unit that activates to the same orientation as the corresponding first
# layer unit responds to images that are optimized for specific features

import matplotlib.pyplot as plt

import Run_dict as Rd
from visualize_parallel import plotting as plot

from datetime import date

# Go through images in folders ConcaveSimple/I_(45, 90, or 135).

class_names = ['IM', ]  # class of image ('apple, orange, etc')
s2 = {'name': [], 'par': {}, 'per': [], 'quart': []}

n_cores = 0
im_size = (256, 256)
J = 5  # number of scales
L = 8  # number of orientations

# List to access the desired folder of features
dirs_I = ['45', '90', '135']
direct = 'ConcaveSimple/I_{}'

# Add to dictionary of arrays - this defines the feature (45, 90, 135), and the associated average activations for
# the second layer unit. For example, in 45: [a, b, c, d, e], a refers to the activation of the second layer unit
# that's selects for the orientation parallel to the first unit in the first layer
orientation_I = {}

f = open("Extracted_Data.txt")
f.write(str(date.today()), '\n')

for feature in dirs_I:
    l_array = Rd.run_parallel(feature, direct, L, J, class_names, s2, n_cores, im_size, orientation_I)
    f.write(str(l_array))

    plot(l_array, L)
    plt.show()

    orientation_I[int(feature)] = l_array
    print(orientation_I)


