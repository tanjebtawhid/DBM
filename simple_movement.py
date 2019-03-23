import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List, Union


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
        """Performs training/iteration of the recurrent network"""
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
    def main(cls):
        """Simulates the network for a given configuration and target"""
        init_config = (0, 0.)
        mov = SimpleMovement(1, 5, 20, True)
        mov.initialize(*init_config)
        mov.set_target(150)
        mov.init_draw()
        mov.move()
        print(mov.alpha, mov.velocity, mov.target)
        plt.show()


class DataGen:

    def __init__(self, height: int, width: int, num_channel: int):
        """Preapres and generates data for autoencoder traiing

        Parameters
        ----------
        height: int
            height of the image
        width: int
            width of the image
        num_channel: int
            number of channels, can be more than 3
        """
        self.height = height
        self.width = width
        self.num_channel = num_channel

    @staticmethod
    def get_target_angels(start: int, end: int, step: int) -> List[int]:
        """
        Generates target angels between start and end(exclusive) according to the
        given step size.

        Parameters
        ----------
        start: int
            starting target angle
        end: int
           largest target angel
        step: int
            diference between target angels

        Returns
        -------
        angels: List[int]
            List of target angels
        """
        angels = []
        for theta in range(start, end, step):
            angels.append(theta)
        return angels

    @staticmethod
    def data_gen(net: SimpleMovement, targets: List[int], init_config: Tuple[int, float], path: str):
        """Iterates over list of targets, peforms simulation for each target, generates and saves
        images in the given directory.

        Parameters
        ----------
        net: SimpleMovement
            instance of MMC network
        targets: List[int]
            list of target angels
        init_config: Tuple[int, float]
            initial alpha and velocity
        path: str
            directory to save images

        Returns
        -------
        """
        if not os.path.exists(path):
            os.mkdir(path)
        for i, target in enumerate(targets):
            data_path = os.path.join(path, str(i))
            if not os.path.exists(data_path):
                os.mkdir(data_path)
            net.initialize(*init_config)
            net.set_target(target)
            net.move(save_dir=data_path)
            np.save(os.path.join(data_path, 'target.npy'),
                    np.array([net.alpha, net.velocity, net.target]).reshape((3, 1)))
            plt.close('all')

    def get_pairs(self, path: str, target_mmc_out: bool, size: Union[int, None]) -> List[List[str]]:
        """Collects filenames of the images as (input, target) pairs. Input can contain more than
        one images depending on the number of channels.

        Parameters
        ----------
        path: str
            path to the directory containing images from MMC simulation
        size: int
            number of images to consider for each target of the MMC network
            simulation. As, at the begining of the simulation of the MMC network
            variation between images are high, size defines the number of iterations
            to take into account.
        target_mmc_out: bool
            whether MMC network output vector or image as target

        Returns
        -------
        ft_pairs: List[List[str]]
            each list containing a (input, target) pair
        """
        ft_pairs = []
        for dirname in os.listdir(path):
            fnames = [fname for fname in os.listdir(os.path.join(path, dirname)) if fname.endswith('.png')]
            size = len(fnames) - 2 if size is None else size
            for j in range(size):
                ft_pairs.append(
                    [os.path.join(path, dirname, fnames[j + i]) for i in range(self.num_channel)])
                if target_mmc_out:
                    ft_pairs[-1] += [os.path.join(path, dirname, 'target.npy')]
                else:
                    ft_pairs[-1] += [os.path.join(path, dirname, fnames[j + self.num_channel])]
        return ft_pairs

    def get_data(self, path: str, target_mmc_out: bool, size: Union[int, None] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generates training data from images produced by the MMC network simulation. Target y can
        be of two different types, i)output vector of the MMC network ii)image based on the MMC output.

        Parameters
        ----------
        path: str
            path to the directory containing images from MMC simulation
        size: int
            number of images to consider for each target of the MMC network
            simulation. As, at the begining of the simulation of the MMC network
            variation between images are high, size defines the number of iterations
            to take into account.
        target_mmc_out: bool
            whether MMC network output vector or image as target

        Returns
        -------
        x: np.ndarray
            images as features
        y: np.ndarray
            either vectors or images
        """
        ft_pairs = self.get_pairs(path, target_mmc_out, size)
        x = np.zeros((len(ft_pairs), self.num_channel, self.height, self.width))
        y = np.zeros((len(ft_pairs), 3)) if target_mmc_out else np.zeros((len(ft_pairs), 1, self.height, self.width))

        for i, item in zip(range(x.shape[0]), ft_pairs):
            for k in range(self.num_channel):
                x[i, k, :, :] = cv2.imread(item[k], 0)
            if target_mmc_out:
                y[i] = np.load(item[-1]).squeeze(axis=1)
            else:
                y[i, 0, :, :] = cv2.imread(item[-1], 0)
        return x, y

    @classmethod
    def main(cls, path: str):
        """Simulates MMC network for different target values.

        Parameters
        ----------
        path: str
            directory to save images from MMC simulation

        Returns
        -------
        """
        init_config = (0, 0.)
        mov = SimpleMovement(1, 5, 20, False)
        targets = DataGen.get_target_angels(15, 190, 15)
        DataGen.data_gen(mov, targets, init_config, path)


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    dpath = 'C:\\Users\\ttanj\\UoB\\WS18\\DBM\\data\\data_simple_movement_2'

    SimpleMovement.main()
    # DataGen.main(dpath)
