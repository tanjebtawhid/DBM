import os
import json
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Union


class SimpleMovement:

    def __init__(self, segment_size: int, beta: int, num_iter: int, live_plot: bool):
        """Simple MMC network with single segment

        Parameters
        ----------
        segment_size: int
            length of the arm segment
        beta: int
            damping factor for velocity
        num_iter: int
            number of iterations
        live_plot: bool
            whether to plot arm movement during simulation
        """
        self.segment_size = segment_size
        self.beta = beta
        self.num_iter = num_iter
        self.live_plot = live_plot
        self.alpha = None
        self.velocity = None
        self.target = None

    def get_weight(self) -> np.ndarray:
        """Initializes the weight matrix of the recuurent network

        Returns
        -------
        w: np.ndarray
            weight matrix
        """
        df = 1 / self.beta
        w = np.array([
            [1, df, 0], [-1, 0, 1], [0, 0, 1]])
        return w

    def initialize(self, alpha: int, velocity: float):
        """Intializes start angel and velocity

        Parameters
        ----------
        alpha: int
            start angel
        velocity: float
            start velocity

        Returns
        -------
        """
        self.alpha = alpha
        self.velocity = velocity

    def set_target(self, target: int):
        """Sets target angel for the segment

        Parameters
        ----------
        target: int
            target angel

        Returns
        -------
        """
        self.target = target

    def set_params(self, alpha: int, velocity: float, target: int):
        """Updates parameters in each iteration"""
        self.alpha = alpha
        self.velocity = velocity
        self.target = target

    def init_draw(self):
        """Initializes figure to plot"""
        plt.figure(figsize=(5, 3))
        target_x = self.segment_size * np.cos(np.radians(self.target))
        target_y = self.segment_size * np.sin(np.radians(self.target))
        self.plot_target = plt.plot([0, target_x], [0, target_y], linestyle='--',
                                    linewidth=1, color='gray', marker='x')[0]
        self.plot_segment = plt.plot([0, target_x], [0, target_y], linestyle='--',
                                     linewidth=2, color='gray', marker='o')[0]
        plt.xlim(-2, 2)
        plt.ylim(-0.5, 2)
        plt.axes().get_yaxis().set_visible(False)
        plt.axes().get_xaxis().set_visible(False)

    def draw(self):
        """Updates plot with current position of the segment"""
        segment_x = self.segment_size * np.cos(np.radians(self.alpha))
        segment_y = self.segment_size * np.sin(np.radians(self.alpha))
        plt.plot([0, segment_x], [0, segment_y], linestyle='--', linewidth=1, color='gray')
        self.plot_segment.set_xdata([0, segment_x])
        self.plot_segment.set_ydata([0, segment_y])

    def save_figure(self, step: int, save_dir: str):
        """Save postion of the segment as image, needed to train the autoencoder

        Parameters
        ----------
        step: int
            current iteration
        save_dir: int
            directory to save image

        Returns
        -------
        """
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.set(xlim=[-3.5, 3.5], ylim=[-1.5, 3.5])
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

        segment_x = self.segment_size * np.cos(np.radians(self.alpha))
        segment_y = self.segment_size * np.sin(np.radians(self.alpha))

        ax.plot([0, segment_x], [0, segment_y], linestyle='-', linewidth=2.0, color='gray')
        fig.savefig(os.path.join(save_dir, (3-len(str(step))) * '0' + str(step) + '.png'))

    def move(self, **kwargs):
        """Performs training/iteration of the MMC network"""
        weights = self.get_weight()
        iteration = 0
        out = np.array([self.alpha, self.velocity, self.target]).reshape((3, 1))
        while iteration < self.num_iter:
            if self.live_plot:
                if iteration % 2 == 0:
                    plt.pause(0.1)
                    self.draw()
            else:
                self.save_figure(iteration, kwargs['save_dir'])
            out = np.dot(weights, out)
            out[-1] = self.target
            self.set_params(*out)
            iteration += 1

    @classmethod
    def simulate(cls, config: Dict[str, Union[int, bool, List]]):
        """Simulates the network for a given configuration and target

        Parameters
        ----------
        config
            network configurations

        Returns
        -------
        """
        net = SimpleMovement(config['segment_length'],
                             config['beta'],
                             config['num_iter'],
                             config['live_plot_net'])
        net.initialize(*config['init_config'])
        net.set_target(config['target'])
        net.init_draw()
        net.move()
        print(net.alpha, net.velocity, net.target)
        plt.show()


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    with open(os.path.abspath('.\\config_simple_movement.json'), 'r', encoding='utf8') as fobj:
        config = json.load(fobj)

    SimpleMovement.simulate(config)
