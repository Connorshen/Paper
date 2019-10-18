import matplotlib.pyplot as plt
import numpy as np
from experiment.static import config
from os import path


class BaseTrainer:
    def __init__(self):
        self.loss_all = []
        self.acc_all = []
        self.step_all = []

    def plot(self):
        if self.net is not None and len(self.acc_all) > 0:
            title_name = self.net.name
            plt.plot(self.step_all, self.acc_all)
            plt.xlabel("step")
            plt.ylabel("acc")
            plt.yticks(np.arange(0, 1.05, 0.1))
            plt.title(title_name)
            plt.savefig(path.join(config.FIG_PATH, title_name + ".png"))
            plt.show()
