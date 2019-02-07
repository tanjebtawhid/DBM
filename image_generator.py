import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement


def compute_coords(theta, seg_len=1):
    x, y = (0, 0)
    ls = [(x, y)]
    for i in range(len(theta)):
        x += seg_len * np.cos(np.radians(sum(theta[:i+1])))
        y += seg_len * np.sin(np.radians(sum(theta[:i+1])))
        ls.append((x, y))
    return ls


def draw(coords):
    fig, ax = plt.figure(figsize=(6, 4))
    xx = [each[0] for each in coords]
    yy = [each[1] for each in coords]
    ax.plot(xx, yy, linestyle='-', linewidth=10.0, color='gray', alpha=0.75, solid_capstyle='round', marker='o')
    ax.set(xlim=[-3.25, 3.25], ylim=[-3.25, 3.25], xticks=[], yticks=[], frame_on=True)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return fig


def img_gen():
    theta_all = np.arange(0, 90, 30)
    for each in combinations_with_replacement(theta_all, 3):
        coords = compute_coords(each)
        fig = draw(coords)
        # fig.savefig('_'.join(str(angle) for angle in each) + '.png')
        plt.show()


if __name__ == '__main__':
    img_gen()
