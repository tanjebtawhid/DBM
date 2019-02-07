import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from sklearn.preprocessing import normalize


class MMC:

    def __init__(self, inputs: Union[None, np.ndarray], df, mode, seg_len, num_iter, live_plot=True):
        self.inputs = inputs
        self.df = df
        self.mode = mode
        self.seg_len = seg_len
        self.num_iter = num_iter
        self.live_plot = live_plot
        self.out = np.zeros((6, 2))

    def __get_weights(self):
        off_diag = 1 / self.df
        diag = (self.df - 2) / self.df
        wmat = np.array([
            [diag, -off_diag, 0, off_diag, -off_diag, off_diag],
            [-off_diag, diag, -off_diag, off_diag, off_diag, 0],
            [0, -off_diag, diag, -off_diag, off_diag, off_diag],
            [off_diag, off_diag, -off_diag, diag, 0, off_diag],
            [-off_diag, off_diag, off_diag, 0, diag, off_diag],
            [off_diag, 0, off_diag, off_diag, off_diag, diag]
        ])
        return wmat

    def set_target(self, target):
        self.inputs = np.array(target)

    def reset(self):
        self.out = np.zeros((6, 2))

    def init_inverse(self, init_config):
        self.out[:3, :] = init_config
        self.out[3, :] = np.sum(init_config[0:2, :], axis=0)
        self.out[4, :] = np.sum(init_config[1:3, :], axis=0)

    def __forward(self):
        self.out[:3, :2] = self.inputs
        weights = self.__get_weights()
        # Suppress recurrent connection of input nodes
        diag_idx = np.diag_indices(3)
        weights[diag_idx] = 0
        iteration = 0
        while iteration < self.num_iter:
            self.out = np.dot(weights, self.out)
            # Enforce input in each iteration
            self.out[:3, :2] = self.inputs
            iteration += 1

    def __inverse(self, **kwargs):
        self.out[-1:, :] = self.inputs
        weights = self.__get_weights()
        # Suppress recurrent connection of input node
        weights[5, 5] = 0
        iteration = 0
        while iteration < self.num_iter:
            if self.live_plot:
                if iteration % 2 == 0:
                    plt.pause(0.1)
                    self.__draw_manipulator()
            else:
                self.__save_arm_image(iteration, kwargs['save_dir'])

            self.out = np.dot(weights, self.out)
            # Enforce input in each iteration
            self.out[-1:, :] = self.inputs
            # Normalize segments length
            segments_normalized = normalize(self.out[:3, :], norm='l2', axis=1)
            self.out[:3, :] = self.seg_len * segments_normalized
            iteration += 1
        plt.close('all')

    def train(self, **kwargs):
        if self.mode == 'forward':
            self.__forward()
        elif self.mode == 'inverse':
            self.__inverse(save_dir=kwargs['save_dir'])

    def initialise_drawing_window(self):
        plt.figure(figsize=(10, 6))
        self.plot_target = plt.plot([0, self.inputs[0]], [0, self.inputs[1]], linestyle=':', linewidth=1.0,
                                    color='gray', marker='x')[0]
        self.plot_arm = plt.plot([0, 0], [0, 0], linestyle='-', linewidth=12.0, color='gray', alpha=0.75,
                                 solid_capstyle='round', marker='o')[0]
        plt.xlim(-3.5, 3.5)
        plt.ylim(-1.5, 3.5)
        plt.axes().get_yaxis().set_visible(False)
        plt.axes().get_xaxis().set_visible(False)

    def __draw_manipulator(self):
        l1 = self.out[0, :]
        l2 = np.sum(self.out[0:2, :], axis=0)
        l3 = np.sum(self.out[0:3, :], axis=0)

        plt.plot([0, l1[0], l2[0], l3[0]], [0, l1[1], l2[1], l3[1]], linestyle='--', linewidth=1.0, color='gray')
        self.plot_arm.set_xdata([0, l1[0], l2[0], l3[0]])
        self.plot_arm.set_ydata([0, l1[1], l2[1], l3[1]])

    def __save_arm_image(self, step, save_dir):
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.set(xlim=[-3.5, 3.5], ylim=[-1.5, 3.5])
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

        l1 = self.out[0, :]
        l2 = np.sum(self.out[0:2, :], axis=0)
        l3 = np.sum(self.out[0:3, :], axis=0)

        ax.plot([0, l1[0], l2[0], l3[0]], [0, l1[1], l2[1], l3[1]],
                linestyle='-', linewidth=2.0, color='gray', solid_capstyle='round')
        fig.savefig(os.path.join(save_dir, (3-len(str(step))) * '0' + str(step) + '.png'))


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    # inputs_forward = np.array([[4, 0], [2, 3], [0, 3]])
    # mmc_net = MMC(inputs_forward, df, 'forward', 0.1)

    mmc_net = MMC(inputs=None, df=10, mode='inverse', seg_len=1, num_iter=50)
    # [0.99, 0.15], [0.707, 0.707], [0.15, 0.99]
    mmc_net.init_inverse(np.array([[0.99, 0.15], [0.707, 0.707], [0.15, 0.99]]))
    mmc_net.set_target([-3., 0.])
    mmc_net.initialise_drawing_window()
    mmc_net.train(save_dir=None)
    print(mmc_net.out)
    plt.show()
