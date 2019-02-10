import os
import re
import cv2
import numpy as np
from mmc_vec import MMC


BASE_PATH = 'C:\\Users\\ttanj\\UoB\\WS18\\DBM\\data\\data_mmc'
HEIGHT = 100
WIDTH = 100
RGB = 1


def get_dirname(target):
    dirname = ''
    for each in target:
        dirname += re.sub(r'\.', '', str(each))
    return os.path.join(BASE_PATH, dirname)


def get_targets():
    targets = []
    for r in range(1, 4):
        for theta in range(0, 190, 30):
            theta = np.radians(theta)
            targets.append([r * np.cos(theta), r * np.sin(theta)])
    return targets


def get_data_pair():
    data_pair = []
    for dirname in os.listdir(BASE_PATH):
        fnames = os.listdir(os.path.join(BASE_PATH, dirname))
        for j in range(len(fnames) - 1):
            data_pair.append((os.path.join(BASE_PATH, dirname, fnames[j]),
                              os.path.join(BASE_PATH, dirname, fnames[j + 1])))
    return data_pair


def prepare_data(num_channel):
    data_pair = get_data_pair()
    x = np.zeros((len(data_pair), HEIGHT * WIDTH * num_channel))
    y = np.zeros((len(data_pair), HEIGHT * WIDTH * num_channel))

    for i, item in zip(range(len(data_pair)), data_pair):
        x[i] = cv2.imread(item[0], RGB).flatten()
        y[i] = cv2.imread(item[1], RGB).flatten()

    return x, y


def data_gen(mmc_net, targets, init_config):
    for i, target in enumerate(targets):
        data_path = os.path.join(BASE_PATH, str(i))
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        mmc_net.init_inverse(init_config)
        mmc_net.set_target(target)
        mmc_net.train(save_dir=data_path)


def main():
    mmc_net = MMC(inputs=None, df=10, mode='inverse', seg_len=1, num_iter=25, live_plot=False)
    init_config = np.array([[0.99, 0.15], [0.707, 0.707], [0.15, 0.99]])
    targets = get_targets()
    data_gen(mmc_net, targets, init_config)


if __name__ == '__main__':
    main()
