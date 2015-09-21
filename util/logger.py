from __future__ import  division
__author__ = 'mmdali'

import logging
import sys

class logger:

    def __init__(self): pass

    def givelogger(self, loggingfilepath):

        log = logging.getLogger('defaultlog')
        log.setLevel(logging.DEBUG)

        #console logger
        consoleformat = logging.Formatter("%(asctime)s:\t%(message)s")
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(consoleformat)
        console.setLevel(logging.INFO)
        log.addHandler(console)

        #file logger
        filelogger = logging.FileHandler(loggingfilepath)
        fileformat = logging.Formatter("%(asctime)s %(levelname)s: \t%(message)s")
        filelogger.setFormatter(fileformat)
        filelogger.setLevel(logging.DEBUG)
        log.addHandler(filelogger)

        return log