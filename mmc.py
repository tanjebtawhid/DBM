import os
import json
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict
from sklearn.preprocessing import normalize


class MMC:

    def __init__(self, seg_len: int, beta: int, num_iter: int, live_plot: bool, mode: str):
        """MMC network with three segmented arm"""
        self.seg_len = seg_len
        self.beta = beta
        self.num_iter = num_iter
        self.live_plot = live_plot
        self.mode = mode
        self.l1 = None
        self.l2 = None
        self.l3 = None
        self.d1 = None
        self.d2 = None
        self.target = None

    def __str__(self) -> str:
        """String representation"""
        return 'l1: {}\nl2: {}\nl3: {}\n' \
               'd1: {}\nd2: {}\ntarget: {}'\
            .format(self.l1, self.l2, self.l3, self.d1, self.d2, self.target)

    def _get_weights(self) -> np.ndarray:
        """Initializes the weight matrix of the recuurent network

        Returns
        -------
        wmat: np.ndarray
            weight matrix
        """
        off_diag = 1 / self.beta
        diag = (self.beta - 2) / self.beta
        wmat = np.array([
            [diag, -off_diag, 0, off_diag, -off_diag, off_diag],
            [-off_diag, diag, -off_diag, off_diag, off_diag, 0],
            [0, -off_diag, diag, -off_diag, off_diag, off_diag],
            [off_diag, off_diag, -off_diag, diag, 0, off_diag],
            [-off_diag, off_diag, off_diag, 0, diag, off_diag],
            [off_diag, 0, off_diag, off_diag, off_diag, diag]
        ])
        return wmat

    def set_target(self, target: List[int]):
        """Sets target coordinate

        Parameters
        ----------
        target: List[int]
            target coordinate
        """
        self.target = np.array(target).reshape(1, 2)

    def get_params(self) -> np.ndarray:
        """Returns network parameters"""
        return np.vstack((self.l1, self.l2, self.l3, self.d1, self.d2, self.target))

    def set_params(self, out: np.ndarray):
        """Sets network parameters

        Parameters
        ----------
        out: np.ndarray
            numpy array containing current values of parameters
        """
        self.d1 = out[3:4, :]
        self.d2 = out[4:5, :]
        if self.mode == 'forward':
            self.target = out[5:6, :]
            return
        self.l1 = out[:1, :]
        self.l2 = out[1:2, :]
        self.l3 = out[2:3, :]

    def initialize(self, init_config: Dict):
        """Initializes network depending on mode of operation

        Parameters
        ----------
        init_config: Dict
            initial network configuration
        """
        if self.mode == 'forward':
            self.init_forward(init_config['forward'])
        else:
            self.init_inverse(init_config['inverse'])

    def init_forward(self, init_config: Dict):
        """Intializes network in forward mode

        Parameters
        ----------
        init_config: Dict
            initial configuration for forward mode
        """
        self.l1 = np.array(init_config['l1']).reshape(1, 2)
        self.l2 = np.array(init_config['l2']).reshape(1, 2)
        self.l3 = np.array(init_config['l3']).reshape(1, 2)
        self.d1 = np.array([[0, 0]])
        self.d2 = np.array([[0, 0]])
        self.target = np.array([[0, 0]])

    def init_inverse(self, init_config: Dict):
        """Initializes network in inverse mode

        Parameters
        ----------
        init_config: Dict
            initial configuration for inverse mode
        """
        self.l1 = np.array(init_config['l1']).reshape(1, 2)
        self.l2 = np.array(init_config['l2']).reshape(1, 2)
        self.l3 = np.array(init_config['l3']).reshape(1, 2)
        self.d1 = np.add(self.l1, self.l2)
        self.d2 = np.add(self.l2, self.l3)

    def _forward(self):
        """forward run"""
        out = np.vstack((self.l1, self.l2, self.l3, self.d1, self.d2, self.target))
        weights = self._get_weights()
        # Suppress recurrent connection of input nodes
        diag_idx = np.diag_indices(3)
        weights[diag_idx] = 0
        iteration = 0
        while iteration < self.num_iter:
            out = np.dot(weights, out)
            # Enforce input in each iteration
            out[:3, :] = np.vstack((self.l1, self.l2, self.l3))
            self.set_params(out)
            iteration += 1

    def _inverse(self, **kwargs):
        """Inverse run

        Parameters
        ----------
        kwargs
            keyword arguments such as figuresize and save directory
        """
        out = np.vstack((self.l1, self.l2, self.l3, self.d1, self.d2, self.target))
        weights = self._get_weights()
        weights[5, 5] = 0
        iteration = 0
        while iteration < self.num_iter:
            if self.live_plot:
                if iteration % 2 == 0:
                    plt.pause(0.1)
                    self._draw()
            else:
                self._save_figure(iteration, kwargs['fig_size'], kwargs['save_dir'])
                plt.close('all')  # for memory issue while generating data
            out = np.dot(weights, out)
            # Enforce input in each iteration
            out[-1:, :] = self.target
            # Normalize segments length
            segments_normalized = normalize(out[:3, :], norm='l2', axis=1)
            out[:3, :] = self.seg_len * segments_normalized
            self.set_params(out)
            iteration += 1

    def init_draw(self):
        """Initializes figure to plot"""
        plt.figure(figsize=(10, 6))
        self.plot_target = plt.plot([0, self.target[0, 0]], [0, self.target[0, 1]], linestyle=':', linewidth=1.0,
                                    color='gray', marker='x')[0]
        self.plot_arm = plt.plot([0, 0], [0, 0], linestyle='-', linewidth=12.0, color='gray', alpha=0.75,
                                 solid_capstyle='round', marker='o')[0]
        plt.xlim(-3.5, 3.5)
        plt.ylim(-1.5, 3.5)
        plt.axes().get_yaxis().set_visible(False)
        plt.axes().get_xaxis().set_visible(False)

    def _draw(self):
        """Updates plot with current position of the segment"""
        l1 = self.l1
        l2 = np.add(self.l1, self.l2)
        l3 = np.add(np.add(self.l1, self.l2), self.l3)

        plt.plot([0, l1[0, 0], l2[0, 0], l3[0, 0]], [0, l1[0, 1], l2[0, 1], l3[0, 1]],
                 linestyle='--', linewidth=1.0, color='gray')
        self.plot_arm.set_xdata([0, l1[0, 0], l2[0, 0], l3[0, 0]])
        self.plot_arm.set_ydata([0, l1[0, 1], l2[0, 1], l3[0, 1]])

    def _save_figure(self, step, fig_size: Tuple[int, int], save_dir: str):
        """Save postion of the segment as image, needed to train the autoencoder

        Parameters
        ----------
        step: int
            current iteration
        fig_size: Tuple[int, int]
            size of the image
        save_dir: str
            directory to save image
        """
        fig, ax = plt.subplots(figsize=fig_size)
        ax.set(xlim=[-3.5, 3.5], ylim=[-1.5, 3.5])
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

        l1 = self.l1
        l2 = np.add(self.l1, self.l2)
        l3 = np.add(np.add(self.l1, self.l2), self.l3)

        ax.plot([0, l1[0, 0], l2[0, 0], l3[0, 0]], [0, l1[0, 1], l2[0, 1], l3[0, 1]],
                linestyle='-', linewidth=2.0, color='gray', solid_capstyle='round')
        fig.savefig(os.path.join(save_dir, (3-len(str(step))) * '0' + str(step) + '.png'))

    def move(self, **kwargs):
        """Performs training/iteration

        Parameters
        ----------
        kwargs
            keyword arguments such as figuresize and save directory
        """
        if self.mode == 'forward':
            self._forward()
        elif self.mode == 'inverse':
            self._inverse(**kwargs)

    @staticmethod
    def simulate(config):
        """Simulates the network for a given configuration and target

        Parameters
        ----------
        config
            network configuration
        """
        net = MMC(config['segment_length'],
                  config['beta'],
                  config['num_iter'],
                  config['live_plot_net'],
                  config['mode'])
        net.initialize(config['init_config'])
        if net.mode == 'inverse':
            net.set_target(config['target'])
            net.init_draw()
        net.move()
        print(net)


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    with open(os.path.abspath('.\\config_mmc.json'), 'r', encoding='utf8') as fobj:
        config = json.load(fobj)

    MMC.simulate(config)
