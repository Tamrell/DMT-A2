import matplotlib.pyplot as plt
import numpy as np
from codebase.io import save_histogram


plt.ion()


class DynamicHistogram():
    #Suppose we know the x range
    min_x = -0.01
    max_x = 1.01
    bin_size = 0.01

    def __init__(self, verbose=True):
        self.verbose = verbose

        self.figure, self.ax = plt.subplots()
        plt.title(f'Distribution of the NDCG scores')
        plt.xlabel("NDCG scores")
        plt.ylabel("Frequency")

        self.bins = np.arange(self.min_x, self.max_x, self.bin_size)
        (_, _, self.bars) = plt.hist([], bins=self.bins)

        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.min_x, self.max_x)
        #Other stuff
        self.ax.grid()

    def update(self, model_id, data, val=False):
        #Update data (with the new _and_ the old points)
        for bar in self.bars:
            bar.remove()
        (_, _, self.bars) = plt.hist(data, bins=self.bins)
        save_histogram(model_id, val=val)

        if not self.verbose:
            return

        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

