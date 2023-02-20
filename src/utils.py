import numpy as np
import matplotlib.pyplot as plt

import config
import os


def plot_and_save(x, y, title, dir_name, save):
    plt.plot(range(len(x)), np.array(y))
    plt.title(title)
    if save:
        os.makedirs(dir_name, exist_ok=True)
        plt.savefig(f"{dir_name}/{title}.png")
        with open(f'{dir_name}/{title}.txt', 'w') as fp:
            fp.write('\n'.join(str(item) for item in y))
    # if config.verbose:
    #     plt.show()
    # else:
    plt.clf()