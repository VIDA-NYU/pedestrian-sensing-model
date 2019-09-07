#!/usr/bin/env python3

import numpy as np

#############################################################
class Person():

    def __init__(self, _id, pos, moveinterval, destiny):
        self.id = _id
        self.destiny = destiny
        self.pos = pos
        self.moveinterval = moveinterval
        self.path = np.array([], dtype=np.int64)
        self.stepstogoal = -1
