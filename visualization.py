import matplotlib.pyplot as plt
import numpy as np


class channel:

    def __init__(self, dict):
        self.dict = dict

    # visualize activity per unit
    def visual(self):
        ind_vis = plt.figure()

        x_array = np.asarray(list(self.dict.keys()))
        y_list = []
        for key in self.dict:
            y_list.append(self.dict[key])
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
        neuron = channel(j_I)
        channel.add_vis(neuron, j1, fig, j1)
    fig.show()


if __name__ == '__main__':
    dict = {45: [0.15257803, 0.27591956, 0.14313947, 0.24798739, 0.19764438, 0.23134904, 0.22275183, 0.13599336],
            90: [0.14930958, 0.1733149, 0.061919015, 0.1328136, 0.2338219, 0.18956801, 0.106374994, 0.0684337],
            135: [0.16952336, 0.17272206, 0.06608763, 0.097738326, 0.24338254, 0.1623848, 0.091786094, 0.06714413]}

    fig, axs = plt.subplots(5)
    visualize_stacked(dict, 5, fig)
    fig.show()
