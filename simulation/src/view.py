#!/usr/bin/env python3

import math
import numpy as np
import utils
import argparse
import os


#############################################################
class View():
    def __init__(self, graph, obstacles, saveplot, log):
        self.mapshape = graph.mapshape

        import matplotlib
        import warnings
        import matplotlib.cbook
        warnings.filterwarnings("ignore",
                                category=matplotlib.cbook.mplDeprecation)
        if saveplot: matplotlib.use('Agg')

        self.show = not saveplot
        import matplotlib.pyplot
        self.plt =  matplotlib.pyplot
        if not saveplot: self.plt.ion()

        self.graph = graph
        self.f, self.axarr = self.plt.subplots(3, figsize=(6, 12), sharex=True)
        self.f.subplots_adjust(hspace=.5)
        self.obstacles = obstacles
        self.log = log

        for i in range(3):
            self.axarr[i].set_xlim(0, graph.mapshape[1])
            self.axarr[i].invert_yaxis()

    def plot_ascii(self, densmap):
        for j in range(self.graph.mapshape[0]):
            for i in range(self.graph.mapshape[1]):
                dens = densmap[j][i]
                if dens == -1:
                    print(' ', end='')
                else :
                    print(dens, end='')
            print()

    def plot_obstacles(self, obstacles, subplot):
        for y, x in obstacles:
            subplot.scatter(x, y, c='gray', marker='s', s=25)

    def plot_points_matplotlib(self, carspos, subplot, col='gray',
                               ptsize=10, marker='s'):
        yy = []
        xx = []

        for p in carspos:
            y, x = np.unravel_index(p, self.graph.mapshape)
            yy.append(y)
            xx.append(x)
        subplot.scatter(xx, yy, c=col, marker=marker, s=ptsize)

        if self.show: self.plt.show()

    def plot_map_matplotlib(self, densmap, _max, subplot, title='', sz=15):
        subplot.clear()
        subplot.set_xlim(0, self.graph.mapshape[1])
        subplot.set_ylim(0, self.graph.mapshape[0])
        subplot.invert_yaxis()
        subplot.set_title(title)
        subplot.axes.get_xaxis().set_visible(False)
        subplot.axes.get_yaxis().set_visible(False)
        yy = []
        xx = []
        cc = []
        maxdens = float(np.max(densmap))
        _color = [0.75, 0.0, 0.0]

        idx = -1

        for dens in densmap:
            idx += 1

            # Not differentiating not seen and zero density spots
            if dens == -1 or dens == 0:
                continue
            else :
                j, i = np.unravel_index(idx, self.graph.mapshape)
                yy.append(j)
                xx.append(i)
                cc.append(_color + [dens/_max])

        npoints = len(xx)
        if npoints == 0: return

        subplot.scatter(xx, yy, c=cc, s=[sz]*npoints, marker='o')

        if self.show: self.plt.show()

    def plot_ascii_densities(self, density1, density2, keyword='out'):
        self.plot_ascii(density1)
        self.plot_ascii(density2)

    def plot_densities(self, density1, density2, density3, carslocations,
                       _max=1.0, tofile=''):
        self.plot_map_matplotlib(density1, _max, self.axarr[0],'True heat map')
        #self.plot_obstacles(self.obstacles, self.axarr[0])
        self.plot_map_matplotlib(density2, _max, self.axarr[1], 'Sensed heat map')
        self.plot_map_matplotlib(density3, _max, self.axarr[2], 'Current heat map')
        self.plot_points_matplotlib(carslocations, self.axarr[2], 'cyan')
        self.plt.tight_layout()

        if self.show:
            self.plt.show()
            self.plt.pause(0.0001)
        else:
            self.plt.savefig(tofile)

#############################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description='Sensing viewer')
    parser.add_argument('indir', help='input dir')
    parser.add_argument('maxcars', help='maximum number of cars')
    parser.add_argument('runs', help='number of runs')
    return parser.parse_args()

#############################################################
def main():
    view = View(searchmap, mapshape, nodes, obstacles, saveplot, log)

if __name__ == '__main__':
    main()
