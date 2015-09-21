from __future__ import  division
__author__ = 'mmdali'

import os
import multiprocessing
import pandas as pnd
from numpy import dot, linalg, array, dot
from itertools import ifilter
from src import draw

def dotfunc(packet):
    return dot(packet[0], packet[1])

def _pertubeuclide(packet):
    samplevec = packet[0]
    cellvec = packet[1]
    smallX = packet[2]
    smallY = packet[3]
    radius = packet[4]
    cellposition = packet[5]
    learnconstant = packet[6]
    cellindex = packet[7]

    sampleposition = array([smallX, smallY])
    distance = linalg.norm(sampleposition-cellposition)
    if distance < radius:
        smoothkernel = (radius - distance)/radius
        output = cellvec + learnconstant * smoothkernel * (samplevec - cellvec)
        return (cellindex, output.tolist())
    else:
        return (cellindex, None)

def _pertubdot(packet):

    samplevec = packet[0]
    cellvec = packet[1]
    smallX = packet[2]
    smallY = packet[3]
    radius = packet[4]
    cellposition = packet[5]
    learnconstant = packet[6]
    cellindex = packet[7]

    sampleposition = array([smallX, smallY])
    distance = linalg.norm(sampleposition-cellposition)
    if distance < radius:
        smoothkernel = (radius - distance)/radius
        output = cellvec + learnconstant * smoothkernel * (samplevec - cellvec)
        return (cellindex, output.tolist())
    else:
        return (cellindex, None)


class training:

    def __init__(self, samplevec_df, cellmap_df, radius_list, learningconstant_list, log=False, excludedcol=False,
                 plot=False, project_dir=False, cpucount = 1):

        self.irr_list = range(len(radius_list))
        self.samplevec_df = samplevec_df
        self.cellmap_df = cellmap_df
        self.radius_list = radius_list
        self.learningconst_list = learningconstant_list
        self.log = log

        if cpucount == -1:
            self.mppool = multiprocessing.Pool()
        else:
            self.mppool = multiprocessing.Pool(cpucount)

        if excludedcol:
            self.excludedcol = ['id', 'x_position', 'y_position', 'pick_count']
            self.excludedcol.extend(excludedcol)
        else:
            self.excludedcol = ['id', 'x_position', 'y_position', 'pick_count']

        self.accpcoll_list = []
        for col in cellmap_df.columns.values:
            if not col in self.excludedcol:
                self.accpcoll_list.append(col)

        if project_dir:
            self.project_dir = project_dir

        if plot:
            self.plot = True
            self.draw_func = draw.draw()
            pertubplotdir = os.path.join(self.project_dir, 'mappertub')
            if os.path.exists(pertubplotdir):
                raise IOError('ERROR: Plot directory already exist:%s' % pertubplotdir)
            os.mkdir(pertubplotdir)
            self.pertubplotdir = pertubplotdir

    def __loginfo(self, msg):
        if self.log:
            self.log.info(msg)

    def __logdebug(self, msg):
        if self.log:
            self.log.debug(msg)

    def givesmallestcell(self, sampleindex, cellmap_df):

        cellmap_list = cellmap_df.as_matrix(columns=self.accpcoll_list).tolist()
        samplevec = array(self.samplevec_df.ix[sampleindex].as_matrix())
        packets = [[array(samplevec, dtype=float), array(item, dtype=float)] for item in cellmap_list]
        dotresult_list = joblib.Parallel(n_jobs=-1)(joblib.delayed(dotfunc(packet) for packet in packets))
        mindiff = min(dotresult_list)
        minindex = dotresult_list.index(mindiff)
        smallestcell = cellmap_df.ix[minindex]
        smallestdict = smallestcell.to_dict()
        smallestdict['magnitude'] = mindiff
        smallestdict['index'] = minindex

        return smallestdict

    def pertubmap(self, smallX, smallY, sampleindex, learnconst, radius, pltname = False, irr = 0):

        packets = []
        for index in self.cellmap_df.index.values:
            internalpack = []
            internalpack.append(self.samplevec_df.ix[sampleindex].as_matrix(columns = self.accpcoll_list))
            internalpack.append(self.cellmap_df.ix[index].as_matrix(columns = self.accpcoll_list))
            internalpack.append(smallX)
            internalpack.append(smallY)
            internalpack.append(radius)
            internalpack.append(self.cellmap_df.ix[index].as_matrix(columns = ['x_position', 'y_position']))
            internalpack.append(learnconst)
            internalpack.append(index)
            packets.append(internalpack)

        pool = multiprocessing.Pool()
        pertubresult = pool.map(_pertubeuclide, packets)
        pool.close()
        pool.join()

        outputdf = pnd.DataFrame(columns=self.accpcoll_list)
        for cellindex, output in pertubresult:
            if not output is None:
                outputdf.loc[cellindex] = output
        for index in outputdf.index.values:
            self.cellmap_df.loc[index, self.accpcoll_list] = outputdf.loc[index, self.accpcoll_list]

        cellmapmatrix = self.cellmap_df.as_matrix(columns=self.accpcoll_list)
        self.cellmap_df['magnitude'] = [linalg.norm(vec) for vec in cellmapmatrix]

        if self.plot:
            figpath = os.path.join(self.pertubplotdir, pltname)
            #self.draw_func.surfaceplt(cellmap_df, figpath, 'magnitude')
            self.draw_func.trainplot(self.cellmap_df, irr, figpath, self.irr_list, self.learningconst_list,
                                     self.radius_list)

    def starttraining(self):

        self.__loginfo('Start training')
        self.cellmap_df['pick_count'] = [0 for i in xrange(len(self.cellmap_df.index.values))]

        for irr, sampleindex, radius, learnconst in zip(self.irr_list, self.samplevec_df.index, self.radius_list,
                                                        self.learningconst_list):

            smallest_dict = self.givesmallestcell(sampleindex, self.cellmap_df)
            self.cellmap_df.loc[smallest_dict['index'], 'pick_count'] += 1
            self.__logdebug('Irr:%s Cell ID: %s X:%s Y:%s' % (irr, smallest_dict['id'], smallest_dict['x_position'],
                                                              smallest_dict['y_position']))
            self.pertubmap(smallest_dict['x_position'],  smallest_dict['y_position'], sampleindex,learnconst,
                           radius, pltname = str(irr), irr=irr)

            if irr % 10 == 0:
                self.__loginfo('Training iteration: %s' % str(irr))

        self.__loginfo('Mapping end')

    def givecellmap(self):

        return self.cellmap_df
