import matplotlib.pyplot as plt
import numpy as np


class channel:

    def __init__(self, unit_index, dict):
        self.index = unit_index
        self.dict = dict

    # visualize activity per unit
    def visual(self):
        ind_vis = plt.figure()

        x_array = np.asarray(list(self.dict.keys()))
        y_list = []
        for key in self.dict:
            y_list.append(self.dict[key][self.index])
        y_array = np.asarray(y_list)

        plot = ind_vis.add_plot(1, 1, 1)
        plt.plot(x_array, y_array)
        plt.title('Unit ' + str(self.index))

        return plot

    # Activity of all units
    def add_vis(self, col_val, tot_vis, ax_y):
        x_array = np.asarray(list(self.dict.keys()))
        y_list = []
        for key in self.dict:
            y_list.append(self.dict[key][self.index])
        y_array = np.asarray(y_list)

        col_val = 'C' + str(col_val)

        axs[ax_y].plot(x_array, y_array, col_val)


def visualize_ind(j_I, J):
    for j1 in range(J):
        neuron = channel(j1, j_I, J)
        plot = channel.visual(neuron)
        plt.show()


def visualize_stacked(j_I, J, fig):
    for j1 in range(J):
        neuron = channel(j1, j_I)
        channel.add_vis(neuron, j1, fig, j1)
    fig.show()


if __name__ == '__main__':
    dice = {1: [1, 2],
            2: [1, 3],
            4: [5, 6]
            }
    fig, axs = plt.subplots(5)
    visualize_stacked(dice, 5, fig)
    fig.show()
