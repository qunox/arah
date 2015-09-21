from __future__ import  division
__author__ = 'mmdali'

import random
import pandas as pnd
import numpy as np
import copy
from os import path
from math import exp
from src import draw

class pretraining:

    def __init__(self):
        self.drawfunc = draw.draw()

    def givetraininglength(self, type, samplesize):
        #'full', 'long', 'short'
        if type == 'short':
            traininglength = 0.3 * samplesize
        elif type == 'full':
            traininglength = samplesize
        elif type == 'long':
            traininglength = 3 * samplesize
        elif type == 'vlong':
            traininglength = 6 * samplesize
        elif isinstance(type, int):
            traininglength = type
        else:
            try:
                traininglength = int(type)
            except ValueError:
                raise Exception('Unknown training length type: %s' , type)

        return int(traininglength)

    def givesamplevec(self , seed_df , samplesize, trainingirrtype, excludedcolumn = None ):

        seedsize = len(seed_df.index.values)

        if trainingirrtype == 'short':
            randowrow = random.sample(seed_df.index, samplesize)
        elif trainingirrtype == 'full':
            randowrow = seed_df.index.values
            random.shuffle(randowrow)
        elif trainingirrtype == 'long':
            randowrow = []
            indexs = seed_df.index.values
            for i in xrange(3):
                shufflingindex = copy.deepcopy(indexs)
                random.shuffle(shufflingindex)
                randowrow.extend(shufflingindex)
        elif trainingirrtype == 'vlong':
            randowrow = []
            indexs = seed_df.index.values
            for i in xrange(6):
                shufflingindex = copy.deepcopy(indexs)
                random.shuffle(shufflingindex)
                randowrow.extend(shufflingindex)
        elif samplesize > seedsize:
            repeatcount = int(samplesize / seedsize)
            excess = samplesize % seedsize
            randowrow = []
            indexs = seed_df.index.values
            for i in xrange(repeatcount):
                shufflingindex = copy.deepcopy(indexs)
                random.shuffle(shufflingindex)
                randowrow.extend(shufflingindex)
            if excess > 0:
                randowrow.extend(random.sample(seed_df.index.values, excess))

        else:
            randowrow = random.sample(seed_df.index, samplesize)

        sample_df = seed_df.ix[randowrow]
        for col in excludedcolumn:
            del sample_df[col]

        return sample_df

    def givelearningconstant(self, traininglenght, k = 'auto', L =1 , x0 = 0.5 , plot = False,projectpath = None,
                             type = 'deriviatetanh'):

        if plot is True and projectpath is None:
            raise Exception('ERROR: Projectpath could not ne None is plot is True')

        midpoint = x0 * traininglenght
        #using logistic function
        irr_list = range(traininglenght)
        if type == 'dampedsin':
            if k == 'auto' : k = 5/traininglenght
            logisticfunc = lambda  x: abs(L * exp(-k*x) * np.cos(0.07 *x))
        elif type == 'logis':
            if k == 'auto':k = 10/traininglenght
            logisticfunc = lambda x: L / (1 + exp(-k*(x - midpoint)))
        elif type == 'reverselogis':
            if k == 'auto':k = 10/traininglenght
            logisticfunc = lambda x: 1 - (L / (1 + exp(-k*(x - midpoint))))
        elif type =='deriviatetanh':
            if k == 'auto':k = 5/traininglenght
            logisticfunc = lambda x: L / (np.cosh((-k*(x - midpoint))) * np.cosh((-k*(x - midpoint))))

        learningconst = map(logisticfunc, irr_list)

        if plot:
            packet = {}
            packet['irr_list'] = irr_list
            packet['learningcontant_list'] = learningconst
            packet['traininglenght'] = traininglenght
            packet['k'] = k
            packet['L'] = L
            packet['midpoint'] = midpoint
            packet['projectpath'] = path.join(projectpath, 'learningconstant.png')
            self.drawfunc.learningconstent(packet)

        return learningconst
    def giveradius(self , traininglenght, initradiuspercent, cellwidth, cellheight , k = 'auto' , L =1 , x0 = 0.5 ,
                   plot = False,projectpath = None ):

        if plot is True and projectpath is None:
            raise Exception('ERROR: Projectpath could not ne None is plot is True')

        if k == 'auto':
            k = 10/traininglenght

        #use logistic function again
        #finding the longest side
        midpoint = x0 * traininglenght
        initradius = initradiuspercent * max([cellheight , cellwidth])
        irr_list = range(traininglenght)
        logisticfunc = lambda x: initradius * (1 - (L / (1 + exp(-k*(x - midpoint)))))
        radiuslist = map(logisticfunc , irr_list)

        if plot:
            packet = {}
            packet['irr_list'] = irr_list
            packet['radiuslist'] = radiuslist
            packet['traininglenght'] = traininglenght
            packet['k'] = k
            packet['L'] = L
            packet['midpoint'] = midpoint
            packet['projectpath'] = path.join(projectpath, 'radiusdecay.png')
            self.drawfunc.radiusdecay(packet)

        return radiuslist

