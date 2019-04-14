import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Tuple


logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(funcName)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


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

    def __str__(self) -> str:
        """String representation"""
        return 'alpha: {}\nvelocity: {}\ntarget: {}'.format(
            self.alpha, self.velocity, self.target)

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

    def initialize(self, init_config: Dict[str, int]):
        """Intializes start angel and velocity

        Parameters
        ----------
        init_config
            initial alpha and velocity
        """
        self.alpha = np.radians(init_config['alpha'])
        self.velocity = init_config['velocity']

    def set_target(self, target: int):
        """Sets target angel for the segment

        Parameters
        ----------
        target: int
            target angel
        """
        self.target = np.radians(target)

    def get_params(self) -> np.ndarray:
        """Returns netowrk parameters"""
        return np.array([self.alpha,
                         self.velocity,
                         self.target]).reshape((3, 1))

    def set_params(self, alpha: int, velocity: float):
        """Updates parameters in each iteration

        Parameters
        ----------
        alpha: int
            current angle of the segment
        velocity: float
            velocity of angular movement
        """
        self.alpha = alpha
        self.velocity = velocity

    def init_draw(self):
        """Initializes figure to plot"""
        plt.figure(figsize=(5, 3))

        target_x = self.segment_size * np.cos(self.target)
        target_y = self.segment_size * np.sin(self.target)

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
        segment_x = self.segment_size * np.cos(self.alpha)
        segment_y = self.segment_size * np.sin(self.alpha)

        plt.plot([0, segment_x], [0, segment_y], linestyle='--', linewidth=1, color='gray')
        self.plot_segment.set_xdata([0, segment_x])
        self.plot_segment.set_ydata([0, segment_y])

    def save_figure(self, step: int, fig_size: Tuple[int, int], save_dir: str):
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

        segment_x = self.segment_size * np.cos(self.alpha)
        segment_y = self.segment_size * np.sin(self.alpha)

        ax.plot([0, segment_x], [0, segment_y], linestyle='-', linewidth=2.0, color='gray')
        fig.savefig(os.path.join(save_dir, (3-len(str(step))) * '0' + str(step) + '.png'))

    def move(self, **kwargs):
        """Performs training/iteration

        Parameters
        ----------
        kwargs
            keyword arguments such as figuresize and save directory
        """
        weights = self.get_weight()
        iteration = 0
        out = np.array([self.alpha, self.velocity, self.target]).reshape((3, 1))
        while iteration < self.num_iter:
            if self.live_plot:
                if iteration % 2 == 0:
                    plt.pause(0.1)
                    self.draw()
            else:
                self.save_figure(iteration, kwargs['fig_size'], kwargs['save_dir'])
                plt.close('all')  # for memory issue while generating data
            out = np.dot(weights, out)
            # out[-1] = self.target
            self.set_params(*out[:2, :])
            iteration += 1

    @staticmethod
    def simulate(config: Dict):
        """Simulates the network for a given configuration and target

        Parameters
        ----------
        config
            network configuration
        """
        net = SimpleMovement(config['segment_length'],
                             config['beta'],
                             config['num_iter'],
                             config['live_plot_net'])
        net.initialize(config['init_config'])
        net.set_target(config['target'])
        net.init_draw()
        net.move()
        logger.info('Network parameters:\n{}'.format(net))


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    with open(os.path.abspath('.\\config_simple_movement.json'), 'r', encoding='utf8') as fobj:
        config = json.load(fobj)

    SimpleMovement.simulate(config)
    plt.show()
