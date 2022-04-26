# Plot the activation of all parallel channels for each feature

import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap

def plotting(dict, L, orientation):
    fig = plt.figure()

    vals = range(L) # label this with orientation
    x_array = [value * (180/L) for value in vals]

    for feature in dict:
        y_array = dict[feature]

    ax = fig.add_subplot(1, 1, 1)
    plt.plot(x_array, y_array, 'ro')

    title = ax.set_title("\n".join(wrap("Activation of 2nd Layer Units Selective for Orientations {} to 1st "
                                        "Layer Unit for Images "
                                        "Optimized for {}".format(orientation, str(feature)))))
    plt.xlabel("Orientation")
    plt.ylabel("Activation")
    plt.grid()

    return fig


if __name__ == '__main__':
    dict = {45: [0.15257803, 0.27591956, 0.14313947, 0.24798739, 0.19764438, 0.23134904, 0.22275183, 0.13599336],
            90: [0.14930958, 0.1733149, 0.061919015, 0.1328136, 0.2338219, 0.18956801, 0.106374994, 0.0684337],
            135: [0.16952336, 0.17272206, 0.06608763, 0.097738326, 0.24338254, 0.1623848, 0.091786094, 0.06714413]}
    L = 8

    for key in dict:
        feature = key
        plotting(dict[feature], L, feature)
        plt.show()
