from __future__ import division
__author__ = 'mmdali'

import os
import json
import pickle
from sys import platform
from datetime import datetime
import os

class config:

    def __init__(self): pass

    def giveconfig(self , configfilepath):

        print '-->Reading configuration file: %s' % configfilepath

        if not os.path.exists(configfilepath):
            raise IOError ('Error: Configuration file %s do not exist' % configfilepath)

        configfile = open(configfilepath, 'r')
        rawconfigdict = json.load(configfile)

        #project setting
        self.loadtrainingcondition = rawconfigdict['loadtrainingcondition']
        if rawconfigdict['loadtrainingcondition'] is True:
            if 'win32' in platform:
                self.initialtrainingconditionpath = rawconfigdict['wininitialtrainingpath']
            else:
                self.initialtrainingconditionpath = rawconfigdict['initialtrainingpath']

        if 'win32' in platform:
            self.projectpath = rawconfigdict['winprojectpath']
            self.sourecepath = rawconfigdict['winsourcepath']
        else:
            self.projectpath = rawconfigdict['projectpath']
            self.sourecepath = rawconfigdict['sourcepath']

        if not os.path.exists(self.projectpath):
            os.mkdir(self.projectpath)

        #logging setting
        self.logfile = os.path.join(self.projectpath , 'arahlog.out')

        #cellmap setting
        self.mapheight = rawconfigdict['mapheight']
        self.mapwidth = rawconfigdict['mapwidth']

        #process setting
        self.processlist = rawconfigdict['process']

        #scalling setting
        self.excludedcol = rawconfigdict['excludedcolumn']
        self.paramlist = rawconfigdict['columtouse']

        #Training setting
        self.usejoblib = rawconfigdict['usejoblib']
        self.trainingiteration_type = rawconfigdict['trainingiteration_type']
        self.gradient = rawconfigdict['gradient']
        self.learnmidpoint = rawconfigdict['learningmidpoint']
        self.radiusmidpoint = rawconfigdict['radiusmidpoint']
        self.learningfunc = rawconfigdict['learning_function']
        self.trainingtype = rawconfigdict['training_type']
        self.logcycle = rawconfigdict['logcycle']
        self.plottraining = rawconfigdict['plotraining']

        #prob_mapping setting
        if 'prob_mapping' in self.processlist:
            self.probmapiter = rawconfigdict['prob_mapping_iter']

        if 'secondsourcepath' in rawconfigdict.keys() or 'winsecondsourcepath' in rawconfigdict.keys():
            self.secondsource = True
            if 'win32' in platform:
                self.secondsourcepath = rawconfigdict['winsecondsourcepath']
            else:
                self.secondsourcepath = rawconfigdict['secondsourcepath']
        else:
            self.secondsource = False

        #DBS setting
        self.dbsmaxdistance = rawconfigdict['dbs_maxdistance']
        self.dbsminimum = rawconfigdict['dbs_minnum']
        self.dbinteractibe = rawconfigdict['dbs_inteactive']

        return self

    def saveconfig(self, overwrite = False):

        configdir = os.path.join(self.projectpath, 'config')
        if not os.path.exists(configdir): os.mkdir(configdir)
        configname = 'arahconfig_'+ datetime.now().strftime('%Y%m%d-%H%M%S') + '.json'
        configfilepath = os.path.join(configdir, configname)

        if os.path.exists(configfilepath) and overwrite is not False:
            raise IOError('Error: Config dump file already exist')
        else:
            dumpfile = open(configfilepath , 'w')
            pickle.dump(self, dumpfile)
            dumpfile.close()

