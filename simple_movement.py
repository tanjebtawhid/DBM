import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


BASE_PATH = 'C:\\Users\\ttanj\\UoB\\WS18\\DBM\\data\\data_simple_movement'


class SimpleMovement:

    def __init__(self, segment_size, beta, num_iter, live_plot):
        self.segment_size = segment_size
        self.beta = beta
        self.num_iter = num_iter
        self.live_plot = live_plot
        self.target = None
        self.out = np.zeros((3, 1))

    def get_weight(self):
        df = 1 / self.beta
        w = np.array([
            [1, df, 0], [-1, 0, 1], [0, 0, 1]])
        return w

    def initialize(self, init_config):
        self.out[:2] = init_config

    def set_target(self, target):
        self.target = target
        self.out[1] = target

    def init_draw(self):
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
        segment_x = self.segment_size * np.cos(np.radians(self.out[0]))
        segment_y = self.segment_size * np.sin(np.radians(self.out[0]))
        plt.plot([0, segment_x], [0, segment_y], linestyle='--', linewidth=1, color='gray')
        self.plot_segment.set_xdata([0, segment_x])
        self.plot_segment.set_ydata([0, segment_y])

    def save_figure(self, step, save_dir):
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.set(xlim=[-3.5, 3.5], ylim=[-1.5, 3.5])
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

        segment_x = self.segment_size * np.cos(np.radians(self.out[0]))
        segment_y = self.segment_size * np.sin(np.radians(self.out[0]))

        ax.plot([0, segment_x], [0, segment_y], linestyle='-', linewidth=2.0, color='gray')
        fig.savefig(os.path.join(save_dir, (3-len(str(step))) * '0' + str(step) + '.png'))

    def move(self, **kwargs):
        weights = self.get_weight()
        iteration = 0
        while iteration < self.num_iter:
            if self.live_plot:
                if iteration % 2 == 0:
                    plt.pause(0.1)
                    self.draw()
            else:
                self.save_figure(iteration, kwargs['save_dir'])
            self.out = np.dot(weights, self.out)
            self.out[-1] = self.target
            iteration += 1

    @classmethod
    def main(cls):
        init_config = np.array([[110], [0]])
        mov = SimpleMovement(1, 5, 20, True)
        mov.initialize(init_config)
        mov.set_target(0)
        mov.init_draw()
        mov.move()
        print(mov.out)
        plt.show()


class DataGen:

    def __init__(self, height, width, num_channel):
        self.height = height
        self.width = width
        self.num_channel = num_channel

    @staticmethod
    def get_target_angles():
        angles = []
        for theta in range(0, 190, 15):
            angles.append(theta)
        return angles

    @staticmethod
    def data_gen(net, targets, init_config):
        for i, target in enumerate(targets):
            data_path = os.path.join(BASE_PATH, str(i))
            if not os.path.exists(data_path):
                os.mkdir(data_path)
            net.initialize(init_config)
            net.set_target(target)
            net.move(save_dir=data_path)
            plt.close('all')

    @staticmethod
    def get_data_pairs():
        data_pairs = []
        for dirname in os.listdir(BASE_PATH):
            fnames = os.listdir(os.path.join(BASE_PATH, dirname))
            for j in range(len(fnames) - 2):
                data_pairs.append((
                    os.path.join(BASE_PATH, dirname, fnames[j]),
                    os.path.join(BASE_PATH, dirname, fnames[j + 1]),
                    os.path.join(BASE_PATH, dirname, fnames[j + 2])
                ))
        return data_pairs

    def prepare_data(self):
        data_pairs = DataGen.get_data_pairs()
        x = np.zeros((len(data_pairs), self.height, self.width, self.num_channel))
        y = np.zeros((len(data_pairs), self.height, self.width, self.num_channel))

        for i, item in zip(range(len(data_pairs)), data_pairs):
            x[i, :, :, 0] = cv2.imread(item[0], 0)
            x[i, :, :, 1] = cv2.imread(item[1], 0)
            y[i, :, :, 0] = cv2.imread(item[2], 0)
            y[i, :, :, 1] = cv2.imread(item[2], 0)

        return x, y

    @classmethod
    def main(cls):
        init_config = np.array([[0], [0]])
        mov = SimpleMovement(1, 5, 20, False)
        targets = DataGen.get_target_angles()
        DataGen.data_gen(mov, targets, init_config)


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    SimpleMovement.main()
    # DataGen.main()
