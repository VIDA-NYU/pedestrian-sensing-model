#!python
#cython: language_level=3, boundscheck=False, wraparound=False
import sys
from heapq import heappush, heappop
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs
cimport cython

MAX = 2147483647 #MAXIMUM INT 32

##########################################################
cpdef get_distance(np.ndarray[np.int64_t, ndim=1] pos1,
                   np.ndarray[np.int64_t, ndim=1] pos2,
                   metric=2):
    """Compute manhattan difference tween @pos1 and @pos2

    Args:
    pos1(tuple): position 1
    pos2(tuple): position 2

    Returns:
    float: manhattan difference
    """
    diffy = fabs(pos1[0]-pos2[0])
    diffx = fabs(pos1[1]-pos2[1])

    if metric == 2:
        distance = sqrt(diffy*diffy + diffx*diffx)
    else:
        distance = fabs(diffy) + fabs(diffx)
    return distance

##########################################################
cpdef recreate_path_from_npy(current, camefrom, mapshape, skipstart=True):
    """Recreate path. In case there is no path, return empty list

    Args:
    current(int): current position
    camefrom(dict): node as key and predecessor as value
    mapshape(tuple): shape of the map
    skiplast(bool): should include @current in the list

    Returns:
    list: positions ordered in the path

    """
    _path = [int(current)]
    v = current

    while v in camefrom.keys():
        v = camefrom[v]
        _path.append(int(v))

    if skipstart: return _path[:-1]
    else: return _path

##########################################################
cpdef get_astar_path(graphstruct, start, goal, initialg):
    """Get A* path

    Args:
    graph(dict): keys are positions and values are lists of neighbours
    s(tuple): starting position
    goal(tuple): goal position
    initialg(dict): dict with eat vertex as keys and max as values.
    If possible, it should be provided for speed sake. Defaults to empty.

    Returns:
    list: list from end to beginning of the path
    """

    cdef np.ndarray[np.int64_t, ndim=2] graph = graphstruct.adj
    cdef np.ndarray[np.int64_t, ndim=1] indices = graphstruct.indices
    cdef np.ndarray[np.int64_t, ndim=2] nodes2d = graphstruct.nodes2d
    cdef np.ndarray[np.int64_t, ndim=1] adjnodes
    cdef np.ndarray[np.int64_t, ndim=1] visitted = np.full(graphstruct.nnodes, 0)
    cdef np.ndarray[np.int64_t, ndim=1] nodes_discovered = np.full(graphstruct.nnodes, 0)
    cdef np.ndarray[double, ndim=1] g
    cdef int current
    cdef int idx
    cdef int nadjnodes
    cdef int v
    cdef double dd = 0.0
    
    mapshape = graphstruct.mapshape

    discovered = []
    camefrom = {}

    g = initialg
    g[:] = MAX

    g[indices[start]] = 0
    sss = nodes2d[indices[start]]
    ggg = nodes2d[indices[goal]]
    startgoalheuristic = get_distance(sss, ggg)
    heappush(discovered, (startgoalheuristic, start))
    nodes_discovered[indices[start]] = 1

    while discovered:
        current = heappop(discovered)[1]
        nodes_discovered[indices[current]] = 0

        if current == goal:
            return recreate_path_from_npy(current, camefrom, mapshape)

        visitted[indices[current]] = 1
        adjnodes = graph[current, :]
                    
        nadjnodes = adjnodes.shape[0]

        for idx in range(nadjnodes):
            v = adjnodes[idx]

            if v < 0:
                if v == -1:
                    dd = 1
                elif v == -2:
                    dd = 1.4142135623730951
                else:
                    break
                continue

            if visitted[indices[v]]: continue

            dist = g[indices[current]] + dd

            if dist >= g[indices[v]]:
                if not nodes_discovered[indices[v]]:
                    heappush(discovered, (MAX, v))
                    nodes_discovered[indices[v]] = 1
                continue

            camefrom[v] = current
            g[indices[v]] = dist

            vvv = nodes2d[indices[v]]
            ggg = nodes2d[indices[goal]]
            neighcost = get_distance(vvv, ggg)

            if not nodes_discovered[indices[v]]:
                heappush(discovered, (neighcost, v))
                nodes_discovered[indices[v]] = 1
    print('Could not find astar path')
    return []

##########################################################
def get_dfs_path(np.ndarray[np.int64_t, ndim=2] graph, mapshape, int start,
                 np.ndarray[np.int64_t, ndim=1] goals):
    """Find the the path between a starting node and a set of goal states. When
    any of the goals is found, the path to it is returned. This algorithm is
    strongly dependent of the the order of the adjacency list provided, it
    always follows the last element of the list.

    Args:
    graph(dict): keys are positions and values are lists of neighbours
    s(tuple): starting position
    goal(tuple): goal position

    Returns:
    list: list from end to beginning of the path
    """

    cdef np.ndarray[np.int64_t, ndim=1] adjnodes
    cdef np.int64_t adjnodes_size
    visitted = set()
    discovered = [start]
    camefrom = {}

    prevnode = []

    if goals[start] == 1: return []

    while discovered:
        current = discovered.pop()
        if current in visitted: continue

        if prevnode != []: camefrom[current] = prevnode

        visitted.add(current)
        prevnode = current

        adjnodes = graph[current]
        adjnodes_size = adjnodes.shape[0]

        for idx in range(adjnodes_size):
            v = adjnodes[idx]

            if v == -1: break

            if goals[v] == 1:
                camefrom[v] = current
                return recreate_path_from_npy(v, camefrom, mapshape)

            discovered.append(v)
    print('DFS could not find path from ' + \
          str(np.unravel_index(start, mapshape)) +  ' to')
    print(goals)

    return []

##########################################################
def update_people_count(np.ndarray[double, ndim=1] curcount,
                        np.ndarray[double, ndim=1] acccount,
                        np.ndarray[long, ndim=1] positions,
                        np.ndarray[long, ndim=1] indices,
                        np.int64_t npeople):
    """Update the current and the accumulated count

    Args:
    curcount(ndarray): numpy flat array containing the current count
    acccount(ndarray): numpy flat array containing the accumulated count

    Returns:
    None
    """
    curcount[:] = 0.0

    cdef int idx
    for idx in range(npeople):
        curcount[indices[positions[idx]]] += 1.0
        acccount[indices[positions[idx]]] += 1.0

##########################################################
def update_car_sensing(np.ndarray[double, ndim=1] curpedscount,
                       np.ndarray[long, ndim=1] nearby,
                       np.ndarray[long, ndim=1] indices,
                       np.ndarray[double, ndim=1] samplesz,
                       np.ndarray[double, ndim=1] pedscount,
                       float sensortpr,
                       float sensornfp):

    cdef int idx
    cdef random_device seed
    cdef binomial_distribution[int] bindist
    cdef poisson_distribution[int] poidist = poisson_distribution[int](sensornfp)
    cdef np.ndarray[long, ndim=1] indicesnearby = indices[nearby]

    for idx in range(nearby.shape[0]):
        samplesz[indicesnearby[idx]] += 1.0
        realcurrcount = curpedscount[indicesnearby[idx]]
        bindist = binomial_distribution[int](realcurrcount, sensortpr)
        pedscount[indicesnearby[idx]] += bindist(seed) + poidist(seed)

##########################################################
cdef extern from '<random>' namespace 'std' nogil:
    cdef cppclass random_device:
        random_device()
        unsigned int operator()()

    cdef cppclass binomial_distribution[T]:
        binomial_distribution(T, double)
        binomial_distribution()
        T operator()[U](U&)

    cdef cppclass uniform_int_distribution[T]:
        uniform_int_distribution(T, int)
        uniform_int_distribution()
        T operator()[U](U&)

    cdef cppclass poisson_distribution[T]:
        poisson_distribution(double)
        poisson_distribution()
        T operator()[G](G&)

##########################################################
cpdef compute_sensed_density(int nnodes,
                             np.ndarray[double, ndim=1] pedscount,
                             np.ndarray[double, ndim=1] samplesz):
    cdef np.ndarray[double, ndim=1] sdens = np.full(nnodes, -1.0)
    cdef int lensamplesz = samplesz.shape[0]
    cdef int i

    for i in range(lensamplesz):
        if samplesz[i] > 0:
            sdens[i] = pedscount[i] / samplesz[i]
        
    return sdens

##########################################################
cpdef compute_density_errors(np.ndarray[double, ndim=1] sdens,
                             np.ndarray[double, ndim=1] tdens):
    """ Compute different error metrics based on the true distribution and
    the measured one

    Args:
    sdens(ndarray): measured distribution of density
    tdens(ndarray): true distribution of density

    Returns:
    list: 3-sized array containing the 1)L1-norm of the difference; 2)L2-norm of
    the difference; 3)L2-norm of the difference of projected measured distribution
    and the true distribution
    """

    cdef int nsensed = 0
    cdef int lensensed = sdens.shape[0]
    cdef double diffl1 = 0.0
    cdef double diffl2 = 0.0
    cdef double stdot = 0.0
    cdef double projerror = 1.0
    cdef double snorm2 = 0.0
    cdef double tnorm2 = 0.0
    cdef double s = 0.0
    cdef double t = 0.0
    cdef double projminustnorm2 = 0.0
    cdef np.ndarray[double, ndim=1] error = np.full(lensensed, -1.0)
    cdef np.ndarray[long  , ndim=1] validid = np.full(lensensed, 1) #mask
    cdef np.ndarray[double, ndim=1] sdensproj = np.full(lensensed, -1.0)
    cdef int i
    cdef int j

    for i in range(lensensed):
        if sdens[i] >= 0:
            validid[nsensed] = i
            nsensed += 1

    if not nsensed: return [-1.0, -1.0, -1.0]

    for j in range(nsensed):
        i = validid[j]
        s = sdens[i]
        t = tdens[i]

        diff = s - t
        diffl1 += fabs(diff)
        diffl2 += diff*diff

        stdot += s*t
        snorm2 += s*s
        tnorm2 += t*t

    diffl1 = diffl1 / lensensed
    diffl2 = sqrt(diffl2) / lensensed

    if snorm2 != 0:
        #Here I am doing a mult by scalar operation
        sdensproj = sdens * (stdot / snorm2)

        for j in range(nsensed):
            i = validid[j]
            diff = sdensproj[i] - tdens[i]
            projminustnorm2 += diff*diff

        scalingfactor = 1 / (sqrt(projminustnorm2) + sqrt(tnorm2))
        projerror = scalingfactor * sqrt(projminustnorm2)

    return [diffl1, diffl2, projerror]

##########################################################
def remove_repeated_trajectories(_path, start, goal):
    if _path.count(start) > 0:
        idx = _path.index(start)
        _path =  _path[:idx]

    if _path.count(goal) > 1:
        idx = len(_path) - list(reversed(_path)).index(goal) - 1
        _path = _path[idx:]
    return _path

##########################################################
def get_wps_path(wp1, wp2, wpspaths):
    """Get cached search between wp1 and wp2

    Args:
    wp1(int): npy indexing of the first waypoint
    wp2(int): npy indexing of the second waypoint

    Returns:
    list: path from wp1 to wp2
    """
    if wp1 == wp2: return []

    cr = sorted([wp1, wp2])
    _path = wpspaths[cr[0]][cr[1]]

    if wp1 != cr[0]: # same order
        _path = list(reversed(_path[1:]))
        _path.insert(0, wp2)

    return _path

##########################################################
cdef norml2(np.ndarray[np.int64_t, ndim=1] v):
    cdef int acc = 0
    cdef int el

    for el in v:
        acc += el*el

    return sqrt(acc)

cpdef get_reachable_2dpoint(A, subt, dist, maxdist):
    # finalpoint = A + len * [(B-A)/norm(B-A)]
    versor = subt / dist
    direc = maxdist*versor
    return (A + direc).astype(int)

##########################################################
cpdef get_closest_wp2d(refpt, waypointskdtree):
    _, inds = waypointskdtree.query(refpt)
    return waypointskdtree.data[inds].astype(int)

##########################################################
def get_closest_wp2d_inside_rect(refpt, ptrect, maxdist, waypoints,
                                 waypointskdtree):

    dists, inds = waypointskdtree.query(refpt, k=10,
                                        distance_upper_bound=maxdist)
    pts2d = []
    for i in inds:
        if i >= len(waypoints): break
        pts2d.append(waypointskdtree.data[i].astype(int))

    idx = get_idx_of_first_point_inside_bbox(pts2d, dists, refpt, ptrect)

    if idx != -1:
        return waypointskdtree.data[inds[idx]].astype(int)
    else:
        return []

##########################################################
def get_tangible_path(start, goal, graph, crossingsmaxdist,
                      wpspaths, g0, waypointskdtree, waypointsidx):

    start2d = graph.nodes2d[graph.indices[start]]
    goal2d = graph.nodes2d[graph.indices[goal]]
    subt = goal2d - start2d
    dist = norml2(subt)

    if dist < crossingsmaxdist:
        return get_path(start, goal, graph, g0, wpspaths, waypointskdtree,
                        waypointsidx)
    else:
        reachablept = get_reachable_2dpoint(start2d, subt, dist,
                                            int(0.9*crossingsmaxdist))

        pt2d = get_closest_wp2d(reachablept, waypointskdtree)

        if pt2d != []:
            return get_path(start,
                                 graph.cachedravel[pt2d[0], pt2d[1]],
                            graph, g0, wpspaths, waypointskdtree, waypointsidx)
        else:
            return get_path(start, goal, graph, g0, wpspaths, waypointskdtree,
                            waypointsidx)

##########################################################
def get_costpoint_tuples(pts, goal2d, cachedravel):
    wps = []
    for pt in pts:
        tup = [ norml2(pt - goal2d), cachedravel[pt[0], pt[1]] ]
        wps.append(np.array(tup))
    return wps

##########################################################
cdef inside_rectangle(p, r1, r2, margin=0):
    if r1[0] <= r2[0]:
        minx = r1[0]
        maxx = r2[0]
    else:
        minx = r2[0]
        maxx = r1[0]

    if r1[1] <= r2[1]:
        miny = r1[1]
        maxy = r2[1]
    else:
        miny = r2[1]
        maxy = r1[1]

    if p[0] >= minx - margin and p[0] <= maxx + margin and  \
            p[1] >= miny - margin and p[1] <= maxy + margin:
        return True
    else:
        return False

##########################################################
cpdef get_idx_of_first_point_inside_bbox(pts, d1, rectpt1, rectpt2, mindist=1):
    for i in range(len(pts)):
        if inside_rectangle(pts[i], rectpt1, rectpt2, 1) and \
                d1[i] >= mindist:
            return i
    return -1

##########################################################
cpdef get_path(start, goal, gr, g0, wpspaths, waypointskdtree,
             waypointsidx, minsmartpath=8):
    """Find a path using the cached paths between waypoints

    Args:
    start(tuple): start
    goal(tuple): goal
    minsmartpath(int): minimum length to compute a smart path.
    If it is less than it, then we just return a flat astar path

    Returns:
    list of tuples: list of paths from end to beginning
    """
    if start == goal: return np.array([])

    start2d = gr.nodes2d[gr.indices[start]]
    goal2d = gr.nodes2d[gr.indices[goal]]

    start2goaldist = norml2(goal2d - start2d)

    if start2goaldist < minsmartpath:
        return np.array(get_astar_path(gr, start, goal, g0))

    swp2d = get_closest_wp2d(start2d, waypointskdtree)
    swp = -1 if swp2d ==[] else gr.cachedravel[swp2d[0], swp2d[1]]
    gwp2d = get_closest_wp2d(goal2d, waypointskdtree)
    gwp = -1 if gwp2d ==[] else gr.cachedravel[gwp2d[0], gwp2d[1]]

    endpath = midpath = stapath = []

    startiswp = waypointsidx[start]
    goaliswp = waypointsidx[goal]

    if startiswp and goaliswp:
        stapath = get_wps_path(start, goal, wpspaths)
    elif startiswp and (not goaliswp) and gwp != -1:
        stapath = get_wps_path(start, gwp, wpspaths)
        midpath = get_astar_path(gr, gwp, goal, g0)
    elif (not startiswp) and goaliswp and swp != -1:
        stapath = get_astar_path(gr, start, swp, g0)
        midpath = get_wps_path(swp, goal, wpspaths)
    elif swp != -1 and gwp != -1:   # not and not
        stapath = get_astar_path(gr, start, swp, g0)
        midpath = get_wps_path(swp, gwp, wpspaths)
        endpath = get_astar_path(gr, gwp, goal, g0)
    else:
        print('not other options')

    fullpath = endpath + midpath + stapath
    if not fullpath:
        pp =  get_astar_path(gr, start, goal, g0)
        #if len(pp) > self.crossingsmaxdist/2:
            #input('computing astar between {} and {}, norm:{}'.format(
                #np.unravel_index(start, gr.mapshape),
                #np.unravel_index(goal, gr.mapshape),
                #norm(np.array(np.unravel_index(start, gr.mapshape)) -
                #np.array(np.unravel_index(goal, gr.mapshape)))
            #))
            #self.maxtillnow = len(pp)
        return np.array(pp)
    else: return np.array(fullpath)

##########################################################
def compute_wps_paths(graph,
                      np.ndarray[np.int64_t, ndim=1] crossings,
                      np.ndarray[double, ndim=1] g0,
                      maxdist=MAX):
    """Get paths of all combinations of waypoints

    Args:
    graph(dict of list): position and respective neighbours
    crossings(list): list of crossings

    Returns:
    dict of lists: tuple of tuple as keys and list (path) as a value
    """

    cdef int i
    cdef int j
    paths = {}
    crossingpaths = {}
    cr = np.sort(crossings)
    ncrossings = len(crossings)

    for i in range(ncrossings):
        crss1 = cr[i]
        crossingpaths[crss1] = {}
        for j in range(i + 1, ncrossings):
            crss2 = cr[j]
            crss1_2d = graph.nodes2d[graph.indices[crss1]]
            crss2_2d = graph.nodes2d[graph.indices[crss2]]

            crssdist = norml2(crss1_2d-crss2_2d)

            if crssdist > maxdist:
                crossingpaths[crss1][crss2] = []
                continue

            finalpath = get_astar_path(graph, crss1, crss2, g0)
            crossingpaths[crss1][crss2] = finalpath
    return crossingpaths

##########################################################
def move_person(int moveinterval,
               int pos,
               int destiny,
               np.ndarray[np.int64_t, ndim=1] path,
               int stepstogoal,
               int tick,
               np.ndarray[np.int64_t, ndim=1] waypoints,
               graph,
               int crossingsmaxdist,
               waypointskdtree,
               np.ndarray[double, ndim=1] g0,
               np.ndarray[np.int64_t, ndim=1] waypointsidx,
               wpspaths):

    cdef random_device seed
    cdef uniform_int_distribution[int] int_dist
    cdef int rndidx

    # For now they are all synchronized
    #if tick % moveinterval != 0: return []

    if stepstogoal < 0:
        # Find new pos
        # Find new destiny
        nwaypoints = len(waypoints)
        pos2d = graph.nodes2d[graph.indices[pos]]
        rad = crossingsmaxdist
        _, inds = waypointskdtree.query(pos2d, k=nwaypoints+1,
                                        distance_upper_bound=rad)
        invalidid = np.where(inds == nwaypoints)[0]

        #TODO: remove it
        #if invalidid.size == 0:
            #raise Exception('Fatal! Increase crossingsmaxdist.')

        int_dist = uniform_int_distribution[int](0, invalidid[0]-1)
        rndidx = int_dist(seed)

        aux = waypointskdtree.data[inds[rndidx]].astype(int)
        destiny = graph.cachedravel[aux[0], aux[1]]
        if destiny != pos:
            path = get_path(pos, destiny, graph, g0, wpspaths,
                              waypointskdtree, waypointsidx)
            stepstogoal = len(path) - 1

    if not path.size or stepstogoal < 0: return []

    pos = path[stepstogoal]
    stepstogoal -= 1
    return (stepstogoal, pos, destiny, path)

##########################################################
def get_cells_in_range(int pos, int rangerad, int maph, int mapw, graph):
    cdef int y
    cdef int x

    y0, x0 = graph.nodes2d[graph.indices[pos]]
    cells = []

    t = y0 - rangerad
    b = y0 + rangerad
    l = x0 - rangerad
    r = x0 + rangerad

    if t < 0: t = 0
    if b >= maph: b = maph - 1
    if l < 0: l = 0
    if r >= mapw: r = mapw - 1

    for y in range(t, b + 1):
        for x in range(l, r + 1):
            if graph.cachedravel[y, x] != -1:
                cells.append(graph.cachedravel[y, x])
    return np.array(cells)
