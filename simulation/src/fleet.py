#!/usr/bin/env python3

import numpy as np

from car import Car
import utils
from utils import MAX
from optimized import update_car_sensing, compute_sensed_density
from optimized import get_cells_in_range

#############################################################
class Fleet():
    def __init__(self, graph, ncars, moveinterval,
                 sensorrad, splinterval, sensortpr, sensornfp, rng,
                 loglevel, wpspaths,
                 maxdist, waypointskdtree, waypointsidx):

        self.log = utils.get_multiprocessing_logger(loglevel)
        self.graph = graph
        self.splinterval = splinterval
        self.sensortpr= sensortpr
        self.sensornfp = sensornfp
        self.moveinterval = moveinterval
        self.lastid = -1
        self.pedscount = np.full(graph.nnodes, 0.0)
        self.samplesz = np.full(graph.nnodes, 0.0) # Later it must be float
        self.sdens = np.full(graph.nnodes, 0.0)
        g0 = np.full(len(graph.nodesflat), MAX, dtype=np.double)

        self.cars = []
        self.insert_cars(ncars, sensorrad, graph.nodesflat, rng, wpspaths, g0,
                         maxdist, waypointskdtree, waypointsidx)

    ##########################################################
    def insert_cars(self, ncars, _range, streets, rng, wpspaths,
                    g0, maxdist, waypointskdtree, waypointsidx):
        freepos = utils.get_random_els_with_reposition(streets, rng, 2*ncars)

        for i in range(ncars):
            curpos = freepos.pop()
            self.lastid += 1

            a = Car(self.lastid, curpos, self.moveinterval,
                    curpos, self.graph.mapshape, wpspaths, maxdist,
                    _range, self.log, self.graph, g0, waypointskdtree,
                    waypointsidx)

            self.cars.append(a)

    ##########################################################
    def update_sensed_density(self):
        self.sdens = compute_sensed_density(self.graph.nnodes,
                                            self.pedscount,
                                            self.samplesz)

    ##########################################################
    def step(self, tick, freepos, pedscount):
        for c in self.cars:
            c.move(tick, freepos)

            if tick % self.splinterval == 0:
                nearby = get_cells_in_range(c.pos,
                                            c.rangerad,
                                            self.graph.mapshape[0],
                                            self.graph.mapshape[1],
                                            self.graph)
                update_car_sensing(pedscount, nearby, self.graph.indices,
                                   self.samplesz, self.pedscount,
                                   self.sensortpr, self.sensornfp)
