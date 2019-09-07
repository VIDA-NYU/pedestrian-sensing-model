#!/usr/bin/env python3

import numpy as np
import math
import random
import time
import scipy.misc
import scipy.signal
import multiprocessing
import json
import itertools
import os
import pprint
from collections import namedtuple
from fractions import gcd
from optimized import get_distance

OBSTACLE = -1
MAX = 2147483647 #MAXIMUM INT 32

Graph = namedtuple('Graph', 'adj nodes2d nodesflat indices cachedravel ' \
                   'mapshape nnodes maplen')

##########################################################
def compute_gcd_intervals(speed1, speed2):
    _gcd = gcd(speed1, speed2)
    interval2 = int(min(speed1, speed2) / _gcd)
    interval1 = int(max(speed1, speed2) / _gcd)
    return interval1, interval2

def get_distance_from_npy_idx(npypos1, npypos2, mapshape):
    """Compute manhattan difference tween @pos1 and @pos2.

    Args:
    pos1(tuple): position 1 in flattened numpy array
    pos2(tuple): position 2 in flattened numpy array

    Returns:
    float: manhattan difference
    """
    pos1 = np.array(np.unravel_index(npypos1, mapshape))
    pos2 = np.array(np.unravel_index(npypos2, mapshape))
    return get_distance(pos1, pos2)

def flatten_indices(indices, mapshape):
    return np.ravel_multi_index(np.transpose(indices), mapshape)

def unflatten_indices(indices, mapshape):
    out = np.unravel_index(indices, mapshape)
    return list(zip(out[0], out[1]))

def parse_image(imagefile, thresh=128):
    """Parse the streets from image and return a numpy ndarray,
    with 0 as streets and OBSTACLE as non-streets. Assumes a 
    BW image as input, with pixels in white representing streets.

    Args:
    imagefile(str): image path

    Returns:
    numpy.ndarray: structure of the image
    """
    img = scipy.misc.imread(imagefile)
    if img.ndim > 2: img = img[:, :, 0]
    return (img > thresh).astype(int) - 1

def find_crossings_crossshape(npmap):
    """Convolve with kernel considering input with
    0 as streets and OBSTACLE as non-streets. Assumes a 
    BW image as input, with pixels in black representing streets.

    Args:
    npmap(numpy.ndarray): ndarray with two dimensions composed of -1 (obstacles)
    and 0 (travesable paths)

    Returns:
    list: set of indices that contains the nodes
    """
    ker = np.array([[0,1,0], [1, 1, 1], [0, 1, 0]])
    convolved = scipy.signal.convolve2d(npmap, ker, mode='same',
                                        boundary='fill', fillvalue=OBSTACLE)
    inds = np.where(convolved >= OBSTACLE)
    return set([ (a,b) for a,b in zip(inds[0], inds[1]) ])

def find_crossings_squareshape(npmap, supressredundant=True):
    """Convolve with kernel considering input with
    0 as streets and -1 as non-streets. Assumes a 
    BW image as input, with pixels in black representing streets.

    Args:
    npmap(numpy.ndarray): ndarray with two dimensions composed of -1 (obstacles)
    and 0 (travesable paths)

    Returns:
    list: set of indices that contains the nodes
    """

    ker = np.array([[1,1], [1, 1]])
    convolved = scipy.signal.convolve2d(npmap, ker, mode='same',
                                        boundary='fill', fillvalue=OBSTACLE)
    inds = np.where(convolved >= 0)
    crossings = np.array([ np.array([a,b]) for a,b in zip(inds[0], inds[1]) ])

    if supressredundant:
        return filter_by_distance(crossings)

    else: return crossings

def filter_by_distance(points, mindist=4):
    """Evaluate the distance between each pair os points in @points 
    and return just the ones with distance gt @mindist

    Args:
    points(set of tuples): set of positions
    mindist(int): minimum distance

    Returns:
    set: set of points with a minimum distance between each other
    """
    cr = list(points)
    npoints = len(points)
    valid = np.full(npoints, np.True_)

    for i in range(npoints):
        if not valid[i]: continue
        for j in range(i + 1, npoints):
            dist = get_distance(cr[i], cr[j])
            if dist < mindist: valid[j] = np.False_

    return points[valid]
    
def get_adjacency_dummy(nodes, npmap):
    return set([ (a,b) for a,b in zip(ind[0], ind[1]) ])

##########################################################
def compute_heuristics(nodes, goal):
    """Compute heuristics based on the adjcency matrix provided and on the goal. If the guy is in the adjmatrix, then it is not an obstacle.
    IMPORTANT: We assume that there is just one connected component.

    Args:
    adjmatrix(dict of list of neighbours): posiitons as keys and neighbours as values
    goal(tuple): goal position
    
    Returns:
    dict of heuristics: heuristic for each position
    """

    subt = np.subtract
    abso = np.absolute
    return {v: np.sum(abso(subt(v, goal))) for v in nodes}

##########################################################
def compute_heuristics_from_map(searchmap, goal):
    s = searchmap

    gy, gx = goal
    height, width = s.shape

    h = {}

    for j in range(height):
        disty = math.fabs(j-gy)
        for i in range(width):
            v = s[j][i]
            if v == OBSTACLE:
                h[(j, i)] = MAX
            else:
                distx = math.fabs(j-gx)
                h[(j, i)] = distx + disty + v
    return h

##########################################################
def get_adjmatrix_from_npy(_map):
    """Easiest approach, considering 1 for each neighbour.
    """

    connectivity = 8
    h, w = _map.shape
    nodes = np.empty((1, 0), dtype=int)
    adj = np.empty((0, 10), dtype=int)

    for j in range(0, h):
        for i in range(0, w):

            if _map[j, i] == OBSTACLE: continue

            nodes = np.append(nodes, np.ravel_multi_index((j, i), _map.shape))
            ns1, ns2 = get_neighbours_coords((j, i), _map.shape)

            neigh[0] = -1
            acc = 1
            neigh = np.full(connectivity, -1)

            for jj, ii in ns1:
                if _map[jj, ii] != OBSTACLE:
                    neigh[acc] = np.ravel_multi_index((jj, ii), _map.shape)
                    acc += 1
            neigh[acc] = -1.4142135623730951 #sqrt(2)
            acc += 1
            adj = np.append(adj, np.reshape(neigh, (1, 10)), axis=0)

    return nodes, adj

##########################################################
def get_full_adjmatrix_from_npy(_mapmatrix):
    """Create a graph structure of a 2d matrix with two possible values: OBSTACLE
    or 0. It returns a big structure in different formats to suit every need

    Returns:
    Structure with attributes
    adj(maplen, 10)  - stores the neighbours of each npy coordinate
    nodes2d(nnodes, 2) - sparse list of nodes in 2d
    nodesflat(nnodes) - sparse list of nodes in npy
    indices(maplen) - dense list of points in sparse indexing
    cachedravel(mapshape) - cached ravel of points to be used
    mapshape(2) - height and width
    nnodes(1) - number of traversable nodes
    """

    h, w = _mapmatrix.shape
    maplen = np.product(_mapmatrix.shape)
    adj = np.full((np.product(_mapmatrix.shape), 10), -1,  dtype=int)
    nodes2d = np.full((maplen, 2), -1, dtype=int)
    nodesflat = np.empty((0, 1), dtype=int)
    indices = np.full(maplen, -1,  dtype=int)
    cachedravel = np.full(_mapmatrix.shape, -1)

    nodesidx = 0
    #TODO: convert everything to numpy indexing
    for j in range(h):
        for i in range(w):

            if _mapmatrix[j, i] == OBSTACLE: continue

            npyidx = np.ravel_multi_index((j, i), _mapmatrix.shape)
            indices[npyidx] = nodesidx

            nodes2d[nodesidx] = np.array([j, i])
            ns1, ns2 = get_neighbours_coords((j, i), _mapmatrix.shape)

            neigh = np.full(10, -MAX)
            neigh[0] = -1
            acc = 1

            cachedravel[j, i] = npyidx
            for jj, ii in ns1:
                if _mapmatrix[jj, ii] != OBSTACLE:
                    neigh[acc] = np.ravel_multi_index((jj, ii), _mapmatrix.shape)
                    acc += 1

            neigh[acc] = -2 #sqrt(2)
            acc += 1

            for jj, ii in ns2:
                if _mapmatrix[jj, ii] != OBSTACLE:
                    neigh[acc] = np.ravel_multi_index((jj, ii), _mapmatrix.shape)
                    acc += 1

            adj[npyidx] = np.reshape(neigh, (1, 10))
            nodesidx += 1

    nodes2d = nodes2d[:nodesidx]
    nodesflat = np.array([ np.ravel_multi_index((xx, yy),_mapmatrix.shape) for xx, yy in nodes2d])
    return Graph(adj=adj, nodes2d=nodes2d, nodesflat=nodesflat,
                 indices=indices, cachedravel=cachedravel,
                 mapshape=_mapmatrix.shape, nnodes=len(nodesflat),
                 maplen=np.product(_mapmatrix.shape))

##########################################################
def get_neighbours_coords(pos, mapshape):
    """ Get neighbours. Do _not_ verify whether it is a valid coordinate

    Args:
    j(int): y coordinate
    i(int): x coordinate
    connectedness(int): how consider the neighbourhood, 4 or 8
    yourself(bool): the point itself is included in the return

    The order returned is:
    5 1 6
    4 9 2
    8 3 7
    """

    j, i = pos
    
    neighbours1 = [  (j-1, i),   (j, i+1),   (j+1, i),   (j, i-1) ]
    neighbours2 = [(j-1, i-1), (j-1, i+1), (j+1, i+1), (j+1, i-1) ]

    n1 = eliminate_nonvalid_coords(neighbours1, mapshape)
    n2 = eliminate_nonvalid_coords(neighbours2, mapshape)

    return (n1, n2)

#########################################################
def get_neighbours_coords_npy_indices(idx, mapshape, connectedness=8,
                                      yourself=False):
    """ Get neighbours. Do _not_ verify whether it is a valid coordinate

    Args:
    idx(int): npy indexing of a matrix
    connectedness(int): how consider the neighbourhood, 8 or 4
    yourself(bool): the point itself is included in the return

    The order returned is:
    c5 c1 c6
    c4 c9 c2
    c8 c3 c7
    """

    nrows, ncols = mapshape
    maplen = np.product(mapshape)
    c1 = idx - ncols
    c2 = idx + 1
    c3 = idx + ncols
    c4 = idx - 1

    neighbours = []
    if c1 >= 0    : neighbours.append(c1)
    if c2 < maplen: neighbours.append(c2)
    if c3 < maplen: neighbours.append(c3)
    if c4 >= 0    : neighbours.append(c4)

    if connectedness == 8:
        c5 = c1 - 1
        c6 = c1 + 1
        c7 = c3 + 1
        c8 = c3 - 1

        if c5 >= 0:
            neighbours.append(c5)
            neighbours.append(c6)
        if c7 < maplen:
            neighbours.append(c7)
            neighbours.append(c8)

    if yourself: neighbours.append(idx)
    return neighbours

##########################################################
def eliminate_nonvalid_coords(coords, mapshape):
    """ Eliminate nonvalid indices

    Args:
    coords(set of tuples): input set of positions
    h(int): height
    w(int): width

    Returns:
    set of valid coordinates
    """

    h, w = mapshape
    valid = []
    for j, i in coords:
        if j < 0 or j >= h: continue
        if i < 0 or i >= w: continue
        valid.append((j, i))

    return valid

##########################################################
def get_adjmatrix_from_image(image):
    """Get the adjacenty matrix from image

    Args:
    searchmap(np.ndarray): our structure of searchmap

    Returns:
    set of tuples: set of the crossing positions
    """

    searchmap = parse_image(image)
    return  get_full_adjmatrix_from_npy(searchmap)

##########################################################
def get_crossings_from_image(imagefile):
    """Get crossings from image file

    Args:
    searchmap(np.ndarray): our structure of searchmap

    Returns:
    set of tuples: set of the crossing positions
    """

    searchmap = parse_image(imagefile)
    return find_crossings_squareshape(searchmap)

##########################################################
def get_obstacles_from_image(imagefile):
    """Get obstacles from image file

    Args:
    searchmap(np.ndarray): our structure of searchmap

    Returns:
    set of tuples: set of the crossing positions
    """

    searchmap = parse_image(imagefile)
    indices = np.where(searchmap==OBSTACLE)
    return set(map(tuple, np.transpose(indices)))

##########################################################
def get_mapshape_from_searchmap(hashtable):
    """Suppose keys have the form (x, y). We want max(x), max(y)
    such that not necessarily the key (max(x), max(y)) exists

    Args:
    hashtable(dict): key-value pairs

    Returns:
    int, int: max values for the keys
    """

    ks = hashtable.keys()
    h = max([y[0] for y in ks])
    w = max([x[1] for x in ks])
    return h+1, w+1

##########################################################
def get_random_els_with_reposition(inputlist, rng, n=1, avoided=[]):
    if not avoided: return [rng.choice(inputlist) for _ in range(n)]

    _list =  list(inputlist)
    nfree = len(_list)
    els = []    # we accept repetitions

    while len(els) < n:
        rndidx = rng.randrange(0, nfree)
        chosen = _list[rndidx]
        if chosen != avoided: els.append(chosen)
    return els

##########################################################
def get_multiprocessing_logger(loglevel):
    log = multiprocessing.log_to_stderr()
    log.setLevel(loglevel)
    return log
##########################################################
def split_all_combinations_from_config(configfile, tmpdir, prefix=''):
    with open(configfile) as fh:
        config = json.load(fh)

    configcopy = []
    _keys = []
    _values = []

    for k, v in config.items():
        if type(v) == list:
            _keys.append(k)
            _values.append(v)

    comb = itertools.product(*_values)

    f = os.path.basename(configfile)
    for c in comb:
        filename = os.path.join(tmpdir, prefix + '_' + (str(c))[1:-1].replace(', ', '-') + '_' + f)

        newconfig = config.copy()
        for i in range(len(c)):
            newconfig[_keys[i]] = [c[i]]

        with open(filename, 'w') as fh:
            json.dump(newconfig, fh)
##########################################################
def copy_list_to_boolsparseindexing(_list, sparseindex):
    boolsparseidx = np.full(sparseindex.shape, np.False_, dtype=np.bool_)
    for el in _list:
        boolsparseidx[el] = True
    return boolsparseidx

##########################################################
def copy_list_to_boolindexing(_list, maplen):
    boolidx = np.full(maplen, 0, dtype=np.int64)
    boolidx[_list] = 1
    return boolidx

##########################################################
def rename_old_folder(filesdir):
    # Unfortunately, it cannot be called from numpy due to the cython file dependency
    # Just create a .py file calling utils.rename_old_folder()

    if not os.path.exists(filesdir):
        print('Dir {} does not exist'.format(filesdir))
        return

    os.chdir(filesdir)

    newnames = {
        'fleetsz':'sensorsnum',
        'rad': 'sensorrange',
        'splng': 'sensorinterval',
        'detprob': 'sensortpr',
        'speed': 'sensorspeed'
    }

    def get_new_set_of_names(params):
        newparams = []
        for param in params:
            p = param
            for k, v in newnames.items():
                if k in p:
                    p = p.replace(k, v)
            newparams.append(p)
        return newparams

    for f in os.listdir('./'):
        if not f.endswith('.npy'): continue
        print(f)
        suff = f.split('.npy')[0]
        params = suff.split('_')
        newparams = get_new_set_of_names(params)
        beg = '_'.join(newparams[:5])
        beg = beg.replace('sensortpr1', 'sensortpr1.0')
        en = '_'.join(newparams[5:])
        newname = '{}_sensorexfp0.0_{}.npy'.format(beg, en)
        print(newname)
        os.rename(f, newname)

