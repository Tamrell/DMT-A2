import matplotlib.pyplot as plt
import numpy as np


plt.ion()


class DynamicHistogram():
    #Suppose we know the x range
    min_x = -0.01
    max_x = 1.01
    bin_size = 0.01

    def __init__(self):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines = plt.hist([], bins=np.arange(self.min_x, self.max_x,
                                                  self.bin_size))
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.min_x, self.max_x)
        #Other stuff
        self.ax.grid()

    def update(self, data):
        #Update data (with the new _and_ the old points)
        plt.hist(data, bins=np.arange(self.min_x, self.max_x, self.bin_size))
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

