#!/usr/bin/python3

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from search import *
from utils import *
from cachedsearch import Cachedsearch
import numpy as np
import random
import scipy
import logging

def test_get_paths_btw_all_wps():
    # _|_|_
    # _|_|_
    #  | |


    ########################################################## test1
    tmpimage = '/tmp/tmp.png'
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


def test_get_path():
    #TODO
    pass

def main():
    test_get_paths_btw_all_wps()
    #test_get_path()

if __name__ == "__main__":
    main()

