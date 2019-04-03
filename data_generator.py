import os
import cv2
import json
import logging
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List, Union, Dict

from simple_movement import SimpleMovement


logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(funcName)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


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
    def get_target_angels(num_segment: int, start: int, end: int, step: int) \
            -> Union[List[int], List[List[float]]]:
        """
        Generates target angels between start and end(exclusive) according to the
        given step size.

        Parameters
        ----------
        num_segment: int
            number of arm segments
        start: int
            starting target angle
        end: int
           largest target angel
        step: int
            diference between target angels

        Returns
        -------
        angels:
            list of target angels of list of target coordinates
        """
        targets = []
        for r in range(1, num_segment+1):
            for theta in range(start, end, step):
                if num_segment == 1:
                    targets.append(theta)
                else:
                    theta = np.radians(theta)
                    targets.append([r * np.cos(theta), r * np.sin(theta)])
        return targets

    @staticmethod
    def data_gen(net: SimpleMovement, targets: List[int], init_config: List[int], path: str):
        """Iterates over list of targets, peforms simulation for each target, generates and saves
        images in the given directory.

        Parameters
        ----------
        net: SimpleMovement
            instance of MMC network
        targets: List[int]
            list of target angels
        init_config: List[int]
            initial alpha and velocity
        path: str
            directory to save images

        Returns
        -------
        """
        if not os.path.exists(path):
            os.mkdir(path)
        for i, target in enumerate(targets):
            logger.info('Current target angle : {} degree'.format(target))
            data_path = os.path.join(path, (3-len(str(i))) * '0' + str(i))
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

    def get_data(self, path: str, target_mmc_out: bool, size: Union[int, None] = None, channel_first: bool = True) \
            -> Tuple[np.ndarray, np.ndarray]:
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
        channel_first: bool
            channel as first dimension or last

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
        if not channel_first:
            x = x.transpose((0, 2, 3, 1))
        return x, y

    @staticmethod
    def generate(path: str, config: Dict[str, Union[int, bool, List[int]]]):
        """Simulates MMC network for different target values.

        Parameters
        ----------
        path: str
            directory to save images from MMC simulation
        config
            network configurations

        Returns
        -------
        """
        mov = SimpleMovement(config['segment_length'],
                             config['beta'],
                             config['num_iter'],
                             config['live_plot_data_gen'])

        targets = DataGen.get_target_angels(config['segment_length'],
                                            config['target_start'],
                                            config['target_stop'],
                                            config['target_step_size'])

        DataGen.data_gen(mov, targets, config['init_config'], path)


if __name__ == '__main__':
    dpath = os.path.abspath('.\\data\\data_simple_movement_2')

    with open(os.path.abspath('.\\config_simple_movement.json'), 'r', encoding='utf8') as fobj:
        config = json.load(fobj)

    DataGen.generate(dpath, config)