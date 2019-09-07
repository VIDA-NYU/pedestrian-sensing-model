#!/usr/bin/env python3

import argparse
import logging
import numpy as np
from numpy import genfromtxt
from multiprocessing import Process, Pool, log_to_stderr
import os
import json
from datetime import datetime
import shutil
#from time import time
import time
from itertools import product
import socket

from model import SensingModel
import utils

#############################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description='Sensing model')
    parser.add_argument('config', nargs='?', default='config/simple.json',
                        help='Config file in json format')
    return parser.parse_args()

#############################################################
def setup_log(verbose):
    lvl = logging.DEBUG if verbose else logging.CRITICAL
    logging.basicConfig(level=lvl,
                        format= '%(asctime)s %(funcName)s: %(message)s',
                        datefmt='%I:%M:%S')
    log = logging.getLogger(__name__)
    return log

#############################################################
def run_one_experiment_given_list(l):
    run_one_experiment(*l)

#############################################################
def parse_config(configfile):
    config = json.load(open(configfile, 'r'))
    if not config['verbose']:
        loglevel = logging.CRITICAL
    elif config['verbose'] == 'debug':
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    log = utils.get_multiprocessing_logger(loglevel)
    log.debug('Experiment parameters:{}'.format(config))

    repeats = range(config['nrepeats'])
    outdir = config['outputdir']
    mapfile = config['map']

    plot = saveplot = False
    if config['view'] == 'plot':
        plot = True
    elif config['view'] == 'file':
        saveplot = True

    k = config['map'].rfind(".")
    cachedpathsfilename = config['map'][:k] + '_paths.pkl'
    peoplespeed = [1]

    npeople = config['npeople']
    sensorsnum = config['sensorsnum']
    sensorrange = config['sensorrange']
    sensorinterval = config['sensorinterval']
    sensortpr = config['sensortpr']
    sensorexfp = config['sensorexfp']
    sensorspeed = config['sensorspeed']
    nticks = [config['nticks']]
    overwrite = [config['overwrite']]
    nprocesses = config['nprocesses']
    maxdist = config['crossingsmaxdist']
                          
    return loglevel, log, repeats, outdir, mapfile, plot,\
        saveplot, cachedpathsfilename, maxdist, peoplespeed, npeople,\
        sensorsnum, sensorrange, sensorinterval, sensortpr, \
        sensorexfp, sensorspeed, nticks, overwrite, nprocesses


#############################################################
def delfile(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

#############################################################
def create_lockfile(lockfile):
    with open(lockfile, 'w') as fh:
        fh.write(socket.gethostname() + '\n')
        fh.write(time.strftime('%Y%m%dT%H%M') + '\n')

#############################################################
def append_str_to_file(lockfile, mystr):
    with open(lockfile, 'a') as fh:
        fh.write(mystr + '\n')

#############################################################
def run_one_experiment(repeatid, npeople, peoplespeed, fleetsz, sensorrad,
                       splnginterval, sensortpr, sensorexfp, fleetspeed, nticks,
                       graph, crossings, cachedpathspkl, crossingsmaxdist,
                       loglevel, outdir, overwrite, viewer=[], savenpy=True):

    t0 = time.time()
    log = utils.get_multiprocessing_logger(loglevel)
    filenamesuffix = 'npeople{}_sensorsnum{}_sensorrange{}_sensorinterval{}' \
        '_sensortpr{}_sensorexfp{}_sensorspeed{}_repeat{}'. \
        format(npeople, fleetsz, sensorrad, splnginterval, sensortpr,
               sensorexfp, fleetspeed, repeatid)
    log.info('Processing ' + filenamesuffix)

    npyfile = os.path.join(outdir, filenamesuffix + '.npy')
    lockfile = os.path.join(outdir, filenamesuffix + '.lock')

    if (os.path.exists(npyfile) or os.path.exists(lockfile)) and not overwrite:
        log.debug('File {} already exists. Skipping'.format(npyfile))
        return

    create_lockfile(lockfile)

    model = SensingModel(graph, crossings, cachedpathspkl, crossingsmaxdist,
                         npeople, peoplespeed, fleetsz, fleetspeed,
                         sensorrad, splnginterval, sensortpr,
                         sensorexfp, loglevel)

    err = np.full((nticks, 3), -1.0)

    for tick in range(nticks):
        model.step()
        err[tick] = model.get_curr_errors()

        if viewer:
            tdens, sdens, curdens = model.get_densities()
            _maxdens = max(np.max(tdens), np.max(sdens), np.max(curdens))
            fleetpos = model.get_fleet_location()

            imagepath = '{}_tick{:004d}.png'.format(filenamesuffix, tick)
            fullfile = os.path.join(outdir, imagepath)

            ttdens = np.full(np.product(graph.mapshape),  0.0)
            ssdens = np.full(np.product(graph.mapshape), -1.0)
            ccurdens = np.full(np.product(graph.mapshape), -1.0)

            ttdens[graph.nodesflat] = tdens
            ssdens[graph.nodesflat] = sdens
            ccurdens[graph.nodesflat] = curdens

            viewer.plot_densities(ttdens, ssdens, ccurdens, fleetpos,
                                  _maxdens, fullfile)

    if savenpy: np.save(npyfile, err)
    elapsed = time.time() - t0
    append_str_to_file(lockfile, str(elapsed))
    log.info('Running {} took {}s'.format(filenamesuffix, elapsed))

#############################################################
def main():
    args = parse_arguments()
    loglevel, log, repeats, outdir, mapfile, plot, saveplot, \
        cachedpathsfilename, crossingsmaxdist, peoplespeed, npeople, \
        sensorsnum, sensorrange, sensorinterval, sensortpr, \
        sensorexfp, sensorspeed, nticks, \
        overwrite, nprocesses =  parse_config(args.config)

    if not os.path.exists(outdir): os.makedirs(outdir)
    shutil.copy(args.config, os.path.join(outdir))

    crossings = utils.get_crossings_from_image(mapfile)
    graph = utils.get_adjmatrix_from_image(mapfile)
    obstacles = utils.get_obstacles_from_image(mapfile)
    log.debug(mapfile + ' loaded.')

    viewer = []

    if plot or saveplot:
        from view import View
        viewer = View(graph, obstacles, saveplot, log)

    params = list(product(repeats, npeople, peoplespeed, sensorsnum,
                          sensorrange, sensorinterval, sensortpr,
                          sensorexfp, sensorspeed, nticks,
                          [graph], [crossings], [cachedpathsfilename],
                          [crossingsmaxdist],
                          [loglevel], [outdir], overwrite, [viewer]))

    if plot or saveplot or nprocesses == 1:
        [ run_one_experiment_given_list(p) for p in params ]
    else:
        pool = Pool(nprocesses)
        pool.map(run_one_experiment_given_list, params)

    log.debug('##########################################################')
    log.debug('Finished.')

#############################################################
if __name__ == '__main__':
    main()
