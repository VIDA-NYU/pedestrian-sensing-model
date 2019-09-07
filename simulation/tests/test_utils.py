#!/usr/bin/python3

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.utils import *
import numpy as np
import random
import scipy

def test_get_manhattan_difference():
    p1 = np.array([ 0, 0])
    p2 = np.array([ 3, 2])
    p3 = np.array([-2, 1])
    assert(get_manhattan_difference(p1, p2) == 5)
    assert(get_manhattan_difference(p1, p3) == 3)
    assert(get_manhattan_difference(p2, p3) == 6)
    assert(get_manhattan_difference(p2, p2) == 0)

def test_parse_image():
    tmpimage = '/tmp/tmp.png'
    streetpos = [(0, 0), (1, 1), (1, 2), (2, 0)]
    blackpx = 128

    mymap1 = np.random.randint(0, blackpx - 50, (3, 3))
    for p in streetpos: mymap1[p] = random.randint(blackpx + 50, 255)
    scipy.misc.imsave(tmpimage, mymap1)

    gndtruthmap = np.full((3, 3), -1)
    for p in streetpos: gndtruthmap[p] = 0

    streets = parse_image(tmpimage, blackpx)
    assert((streets == gndtruthmap).all())

    mymap2 = np.full((3, 3), 0)
    scipy.misc.imsave(tmpimage, mymap2)
    streets2 = parse_image(tmpimage, blackpx)
    assert((np.full((3, 3), -1) == streets2).all())

def test_find_crossings_crossshape():
    streetspos = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
    mymap1 = np.full((3, 3), -1)
    for p in streetspos: mymap1[p] = 0
    crossings = find_crossings_crossshape(mymap1)
    assert(set([(1, 1)]) == crossings)

    mymap2 = np.full((3, 3), -1)
    crossings2 = find_crossings_crossshape(mymap2)
    assert(set() == crossings2)

def test_find_crossings_squareshape():
    streetspos = [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    mymap1 = np.full((3, 3), -1)
    for p in streetspos: mymap1[p] = 0
    crossings = find_crossings_squareshape(mymap1)
    assert(np.array_equal([(2, 1)], crossings))

    mymap2 = np.full((3, 3), 0)
    crossings2 = find_crossings_squareshape(mymap2, True)
    assert(1 == len(crossings2))

def test_get_neighbours_coords_npy_indices():
    mapshape = (4, 4)
    idx = 13

    assert(get_neighbours_coords_npy_indices(idx, mapshape, connectedness=4,
                                             yourself=False) == [9, 14, 12])
    assert(get_neighbours_coords_npy_indices(idx, mapshape, connectedness=8,
                                             yourself=False) == [9, 14, 12, 8, 10])
    assert(get_neighbours_coords_npy_indices(idx, mapshape, connectedness=8,
                                             yourself=True) == [9, 14, 12, 8, 10, 13])

def test_get_neighbours_coords():
    mapshape = (3, 3)
    idx0 = (1, 1)
    coords0 = [(0, 1), (1, 2), (2, 1), (1, 0), \
              (0, 0), (0, 2), (2, 2), (2, 0)]

    assert(get_neighbours_coords(idx0, mapshape, connectedness=8,
                                 yourself=False) == coords0)

    coords1 = coords0 + [idx0]
    assert(get_neighbours_coords(idx0, mapshape, connectedness=8,
                                 yourself=True) == coords1)

    coords2 = [(0, 1), (1, 2), (2, 1), (1, 0)]
    assert(get_neighbours_coords(idx0, mapshape, connectedness=4,
                                 yourself=False) == coords2)

    idx3 = (2, 2)
    coords3 = [(1, 2), (2, 1), (1, 1)]
    assert(get_neighbours_coords(idx3, mapshape, connectedness=8,
                                 yourself=False) == coords3)

def test_get_adjmatrix_from_image():
    tmpimage = '/tmp/tmp.png'
    streetpos = [(0, 0), (1, 1), (1, 2), (2, 0)]
    nnodes = len(streetpos)
    blackpx = 128
    mapshape = (3, 3)
    maplen = np.product(mapshape)
    mymap1 = np.random.randint(0, blackpx - 50, mapshape)
    for p in streetpos: mymap1[p] = random.randint(blackpx + 50, 255)
    scipy.misc.imsave(tmpimage, mymap1)
    adj = np.full((maplen, 8), -1)
    adj[0, 0:1] = 4
    adj[4, 0:3] = [5, 0, 6]
    adj[5, 0:1] = 4
    adj[6, 0:1] = 4
    nodes2d = streetpos
    nodesflat = [0, 4, 5, 6]
    indices = np.full(maplen, -1)

    cachedravel = np.full(mapshape, -1)
    for p in nodes2d:
        cachedravel[p[0], p[1]] = np.ravel_multi_index(p, mapshape)
    graph = Graph(adj=adj, nodes2d=nodes2d, nodesflat=nodesflat,
                  indices=indices, mapshape=mapshape,
                  nnodes=len(nodesflat), cachedravel=cachedravel)
    
    computed = get_adjmatrix_from_image(tmpimage)
    assert(np.array_equal(graph.adj, computed.adj))
    assert(np.array_equal(graph.nodes2d, computed.nodes2d))
    assert(np.array_equal(graph.nodesflat, computed.nodesflat))
    assert(np.array_equal(graph.cachedravel, computed.cachedravel))

def test_copy_list_to_boolindexing():
    els1 = [0,3,4]
    maplen = 5
    assert(np.array_equal(copy_list_to_boolindexing(els1, maplen),
                          np.array([1, 0, 0, 1, 1]).astype(bool)))

    els2 = []
    assert(np.array_equal(copy_list_to_boolindexing(els2, maplen),
                          np.full(maplen, False)))
    els3 = list(range(maplen))
    assert(np.array_equal(copy_list_to_boolindexing(els3, maplen),
                          np.full(maplen, True)))

def main():
    test_get_manhattan_difference()
    test_parse_image()
    test_find_crossings_crossshape()
    test_find_crossings_squareshape()
    test_get_neighbours_coords_npy_indices()
    test_get_neighbours_coords()
    test_get_adjmatrix_from_image()
    test_copy_list_to_boolindexing()

if __name__ == "__main__":
    main()

