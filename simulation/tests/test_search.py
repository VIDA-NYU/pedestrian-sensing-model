#!/usr/bin/python3

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from search import *
from utils import *
import numpy as np
import random
import scipy

def test_recreate_path():
    current = (0, 0)
    camefrom = {(0, 0):(0, 1),
                (0, 1):(0, 2),
                (0, 2):(0, 3)}
   
    _path = [ (0, 0), (0, 1), (0, 2), (0,3) ]
    assert(recreate_path(current, camefrom, True) == _path[:-1])
    assert(recreate_path(current, camefrom, False) == _path)
    assert(recreate_path((0, 4), camefrom, True) == [])

def test_get_astar_path():
    tmpimage = '/tmp/tmp.png'

    ########################################################## test1
    im1 = np.array([[1, 0, 0],
                    [0, 1, 1],
                    [0, 1, 1]])

    scipy.misc.imsave(tmpimage, im1)
    graph1 = get_adjmatrix_from_image(tmpimage)
    initialg = np.full(len(graph1.nodesflat), MAX)
    assert(get_astar_path(graph1, 0, 5, initialg) == [5, 4])
    assert(get_astar_path(graph1, 0, 0, initialg) == [])
    assert(get_astar_path(graph1, 0, 4, initialg) == [4])
    assert(get_astar_path(graph1, 0, 8, initialg) == [8, 4])

    ########################################################## test1
    im2 = np.array([[1, 1, 0, 0],
                    [1, 0, 0, 0],
                    [1, 1, 1, 0],
                    [1, 0, 0, 0],
                    [1, 0, 1, 1],
                    [0, 1, 1, 1]])

    scipy.misc.imsave(tmpimage, im2)
    graph2 = get_adjmatrix_from_image(tmpimage)
    initialg = np.full(len(graph2.nodesflat), MAX)

    assert(get_astar_path(graph2, 0, 1, initialg) == [1])
    assert(get_astar_path(graph2, 0, 0, initialg) == [])

    # TODO: Below fails! Add less weight to diagonal paths
    #assert(get_astar_path(graph2, 0, 23) == [23, 22, 21, 16, 12, 8, 4])

    assert(get_astar_path(graph2, 0, 10, initialg) == [10, 9, 4])


##########################################################
def test_get_dfs_path():
    tmpimage = '/tmp/tmp.png'

    ########################################################## test1
    im1 = np.array([[1, 0, 0],
                    [0, 1, 1],
                    [0, 1, 1]])

    scipy.misc.imsave(tmpimage, im1)
    graph1 = get_adjmatrix_from_image(tmpimage)
    crossings = utils.get_crossings_from_image(tmpimage)
    start1 = 0
    flatwaypoints = utils.flatten_indices(crossings, graph1.mapshape)
    waypoints = copy_list_to_boolindexing(flatwaypoints, len(graph1.adj)).astype(np.int64)
    assert(get_dfs_path(graph1.adj, graph1.mapshape, 0, waypoints) == [8, 4])
    assert(get_dfs_path(graph1.adj, graph1.mapshape, 4, waypoints) == [8])
    assert(get_dfs_path(graph1.adj, graph1.mapshape, 8, waypoints) == [])

    ########################################################## test1
    im2 = np.array([[1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 1, 1],
                    [0, 1, 1, 1]])

    scipy.misc.imsave(tmpimage, im2)
    graph2 = get_adjmatrix_from_image(tmpimage)
    crossings2 = utils.get_crossings_from_image(tmpimage)
    start2 = 0
    flatwaypoints2 = utils.flatten_indices(crossings2, graph2.mapshape)
    waypoints2 = copy_list_to_boolindexing(flatwaypoints2, len(graph2.adj)).astype(np.int64)

    assert(get_dfs_path(graph2.adj, graph2.mapshape, 0, waypoints2) == [5])
    waypoints3 = waypoints2
    waypoints3[5] = 0
    assert(get_dfs_path(graph2.adj, graph2.mapshape, 0, waypoints3) == [23, 18, 21, 16, 12, 8, 5])

def main():
    test_get_astar_path()
    test_get_dfs_path()

if __name__ == "__main__":
    main()

