#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class LambdaPlotter():
    """
    lambda value plotter
    """
    def __init__(self, t):
        # figure and axes for time intensity plot
        self.fig = plt.figure()
        self.ax  = self.fig.add_subplot(111)
        self.t   = t
        plt.ion()

    def update(self, sim):
        imshow_frame_ind = 3
        if len(sim.shape) == 1:
            print("plotting temporal lambda")
            # clear last figure
            self.ax.clear()
            self.ax.plot(self.x, sim, label="Simulation")
            self.ax.set_xlim([0., 1.])
            self.ax.set(xlabel="t", ylabel="lambda")
            plt.pause(0.02)
        elif len(sim.shape) == 3:
            # sim [n_tgrid, n_sgrid, n_sgrid]
            print("plotting spatial lambda")
            # clear last figure
            for frame_id in range(sim.shape[0]): 
                self.ax.clear()
                self.ax.imshow(sim[frame_ind, :, :])
                plt.pause(0.2)