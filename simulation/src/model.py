#!/usr/bin/env python3

import numpy as np
import os
import random
from time import time
import logging
import pickle
from scipy.spatial import cKDTree

from person import Person
from fleet import Fleet
from utils import get_random_els_with_reposition, MAX, copy_list_to_boolindexing
from utils import get_multiprocessing_logger, compute_gcd_intervals
from optimized import update_people_count, compute_density_errors
from optimized import compute_wps_paths, get_path, move_person

#############################################################
class SensingModel():
    def __init__(s, graph, waypoints, pathsfile, maxdist, npeople,
                 peoplespeed, ncars, fleetspeed, sensorrad, splinterval,
                 sensortpr, sensornfp, loglevel):

        s.log = get_multiprocessing_logger(loglevel)
        s.graph = graph
        wpaux = np.transpose(waypoints)
        s.waypoints = graph.cachedravel[wpaux[0], wpaux[1]]
        s.g0 = np.full(graph.nnodes, MAX, dtype=np.double)
        s.wpspaths = s.load_cached_wps_paths(pathsfile, graph, maxdist,
                                             s.waypoints, s.g0)
        s.crossingsmaxdist = maxdist
        s.npeople = npeople
        s.acccount = np.full(graph.nnodes, 0.0)
        s.curcount = np.full(graph.nnodes, 0.0)
        s.clock = 0
        s.lastid = -1

        s.rng = random.SystemRandom()

        s.waypointsidx = copy_list_to_boolindexing(s.waypoints, graph.maplen)
        s.waypointskdtree = cKDTree(waypoints, copy_data=False)

        s.people = []

        pintvl, fintvl =  compute_gcd_intervals(peoplespeed, fleetspeed)
        s.insert_people(npeople, pintvl, waypoints)
        s.fleet = Fleet(graph, ncars, fintvl, sensorrad, splinterval,
                        sensortpr, sensornfp, s.rng, loglevel,
                        s.wpspaths,
                        s.crossingsmaxdist, s.waypointskdtree, s.waypointsidx)

        s.tdens = np.full(graph.nnodes, 0.0)
        s.denserrors = np.array([-1,-1,-1])

    ##########################################################
    def load_cached_wps_paths(s, filename, graph, crossingsmaxdist,
                                   waypoints, g0):

        if os.path.isfile(filename):
            s.log.debug('Loading cached crossings paths from file ' + \
                           filename)
            wpspaths = pickle.load(open(filename, 'rb'))
        else:
            s.log.debug('Computing crossings paths and saving them ' + \
                           'to {}.'.format(filename))
            wpspaths = compute_wps_paths(graph, waypoints, g0,
                                                  1.2*crossingsmaxdist)
            pickle.dump(wpspaths, open(filename, 'wb'))
        return wpspaths

    ##########################################################
    def insert_people(s, npeople, moveintvl, crossings):
        freepos = get_random_els_with_reposition(s.graph.nodesflat,
                                                 s.rng, 2*npeople)

        indices = s.graph.indices
        for i in range(npeople):
            start = freepos.pop()

            s.lastid += 1
            a = Person(s.lastid, start, moveintvl, start)

            s.people.append(a)

            s.acccount[indices[start]] += 1

    ##########################################################
    def update_true_density(s):
        if s.clock == 0:
            s.tdens = np.full(s.graph.nnodes, -1.0)
        else:
            s.tdens = s.acccount / float(s.clock)

    ##########################################################
    def update_densities(s):
        s.update_true_density()
        s.fleet.update_sensed_density()
        s.denserrors = compute_density_errors(s.fleet.sdens, s.tdens)

    ##########################################################
    def get_fleet_location(self):
        return [ c.pos for c in self.fleet.cars ]

    ##########################################################
    def get_densities(s):
        return s.tdens, s.fleet.sdens, s.curcount

    ##########################################################
    def get_curr_errors(s):
        return s.denserrors

    ##########################################################
    def create_path(s, ag):
        if ag.destiny == ag.pos: return []
        ag.path = get_path(ag.pos, ag.destiny)
        ag.stepstogoal = len(ag.path) - 1

        if not ag.path:
            log.warning('Could not find path from {} to {}'. \
                     format(ag.pos, ag.destiny))

    ##########################################################
    def step(s):
        s.clock += 1

        nnewpaths = 0
        for ag in (s.fleet.cars + s.people):
            if ag.stepstogoal < 0: nnewpaths += 1

        freepos = get_random_els_with_reposition(s.graph.nodesflat,
                                                 s.rng, nnewpaths)

        if s.clock % s.people[0].moveinterval == 0:
            for p in s.people:
                pstruct = move_person(p.moveinterval, p.pos, p.destiny, p.path,
                                      p.stepstogoal, s.clock, s.waypoints,
                                      s.graph, s.crossingsmaxdist,
                                      s.waypointskdtree, s.g0,
                                      s.waypointsidx, s.wpspaths)

                if pstruct: p.stepstogoal, p.pos, p.destiny, p.path = pstruct

        peoplepositions = np.array([ p.pos for p in s.people])
        update_people_count(s.curcount, s.acccount, peoplepositions,
                            s.graph.indices, s.npeople)
        s.fleet.step(s.clock, freepos, s.curcount)
        s.update_densities()
