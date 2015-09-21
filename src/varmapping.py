from __future__ import division
__author__ = 'mmdali'


import os
import multiprocessing
from src import draw, dfutil
from copy import deepcopy
from scipy.spatial import distance as dt
from numpy import array_split, mean, std, argmin


def _maptosmallest(jobpacket):
    cellmap_df = jobpacket[0]
    paramlist = jobpacket[1]
    minisource_df = jobpacket[2]
    distancetype = jobpacket[3]

    resultlist = []

    minisource_matrix = minisource_df.as_matrix(columns=paramlist)
    cellmap_matrix = cellmap_df.as_matrix(columns=paramlist)
    distance_matrix = dt.cdist(minisource_matrix, cellmap_matrix, metric=distancetype)
    cellmapdfindex_list = cellmap_df.index.values
    minisourceindex_list = minisource_df.index.values

    for index in range(len(minisource_matrix)):

        distances = distance_matrix[index]
        minindex = argmin(distances)
        mincelldfindex = cellmapdfindex_list[minindex]
        minsourcedfindex = minisourceindex_list[index]

        resultlist.append([mincelldfindex, minisource_df.loc[minsourcedfindex, 'id'],
                           minisource_df.loc[minsourcedfindex, 'label']])

    return resultlist

class mapping:
    def __init__(self, cellmap_df, normalizedsource_df, projectpath, paramlist, type, normalize = True ,plot=False, log=False):

        self.cellmap_df = cellmap_df
        self.source_df = normalizedsource_df
        self.projectpath = projectpath

        self.paramlist = paramlist


        if type not in ['euclidean','cityblock', 'correlation',  'chebyshev', 'cosine', 'canberra', 'braycurtis', 'mahalanobis']:
            raise Exception('Unknown mapping type: %s' % type)
        else:
            self.trainingtype = type

        if plot:
            self.plot = True
            self.draw_func = draw.draw()
            self.drawingdir = os.path.join(projectpath, 'mapcountour')

            if not os.path.exists(self.drawingdir):
                os.mkdir(self.drawingdir)
        else:
            self.plot = False

        self.normalize = normalize
        if normalize is True:
            self.dfutil_func = dfutil.dfutil()

        if log:
            self.log = log
        else:
            self.log = False


    def __loginfo(self, msg):
        if self.log: self.log.info(msg)

    def __logdebug(self, msg):
        if self.log: self.log.debug(msg)

    def __posprocessing(self, name, mapresult_list):

        cellmeplen = len(self.cellmap_df)
        self.cellmap_df[name +'source_id'] = [[] for i in range(cellmeplen)]
        self.cellmap_df[name +'source_label'] = [[] for i in range(cellmeplen)]

        for cellindex, sourceid, sourcelabel in mapresult_list:

            self.cellmap_df.ix[cellindex][name +'source_id'].append(sourceid)
            self.cellmap_df.ix[cellindex][name +'source_label'].append(sourcelabel)

        sourcecount_list = []
        signalcount_list = []
        noisesecount_list = []

        for index in self.cellmap_df.index.values:
            sourcecount_list.append(len(self.cellmap_df.loc[index, name +'source_id']))
            signalcount_list.append(self.cellmap_df.loc[index, name +'source_label'].count(1.0))
            noisesecount_list.append(self.cellmap_df.loc[index, name +'source_label'].count(0.0))


        self.cellmap_df[name +'source_count'] = sourcecount_list
        self.cellmap_df[name +'signal_count'] = signalcount_list
        self.cellmap_df[name +'noise_count'] = noisesecount_list

        countmean = mean(sourcecount_list)
        countstd = std(sourcecount_list)
        excess_list =[]
        for index in self.cellmap_df.index.values:
            sourcecount = self.cellmap_df.loc[index, name +'source_count']
            excess_list.append((sourcecount - countmean) / countstd)
        self.cellmap_df[name +'excess_ratio'] = excess_list

        self.__loginfo('Saving mapped cell map')
        mappedcellmap_filepath = os.path.join(self.projectpath, 'mappedcell.csv')
        self.cellmap_df.to_csv(mappedcellmap_filepath, index_label=False, index=False)

        if self.plot:
            self.__loginfo('Making contour plots: pid=%s')
            sourcecount_plt = os.path.join(self.drawingdir, name +'source_count_map.png')
            self.draw_func.contourfplot(self.cellmap_df, sourcecount_plt, name +'source_count',
                                       title=name +'Event count distribution')

            self.__loginfo('Making surface plots plots')
            sourcecount_plt = os.path.join(self.drawingdir, name +'source_count_surfface.png')
            self.draw_func.surfaceplt(self.cellmap_df, sourcecount_plt, name +'source_count',
                                      title=name +'Event count distribution')

            sourcecount_plt = os.path.join(self.drawingdir, name +'excess_ratio_map.png')
            self.draw_func.contourfplot(self.cellmap_df, sourcecount_plt, name +'excess_ratio',
                                       title='%s Excess Ratio\nMean=%s Std=%s' % (name, countmean, countstd))

            self.__loginfo('Making surface plots')
            sourcecount_plt = os.path.join(self.drawingdir, name +'excess_ratio_surfface.png')
            self.draw_func.surfaceplt(self.cellmap_df, sourcecount_plt, name +'excess_ratio',
                                      title='%s Excess Ratio\nMean=%s Std=%s' % (name, countmean, countstd))

            self.__logdebug('Finish making contour plot')


    def mapsource(self, title, cpucount=None):

        if self.normalize is False:
            self.__loginfo('WARNING UN-NORMALIZE MAPPING!!')
        else:
            self.__loginfo('Normalizing the source')
            self.source_df, scaller = self.dfutil_func.stdscaler(self.source_df, self.paramlist)
            self.__loginfo('Normalzing the cellmapp')
            self.cellmap_df, _ = self.dfutil_func.stdscaler(self.cellmap_df, self.paramlist, scaller)

        if cpucount is None:
            cpucount = multiprocessing.cpu_count() - 1
        self.__logdebug('Dividing the source data frame')

        #deviding the source by the number of process spawned
        minisourcedflist = array_split(self.source_df, cpucount)

        #creating job packets
        jobpacket = [(deepcopy(self.cellmap_df), deepcopy(self.paramlist), deepcopy(minidf), deepcopy(self.trainingtype))  for minidf in minisourcedflist ]

        self.__logdebug('Creating multiprocessing pool')
        pool = multiprocessing.Pool(cpucount)
        self.__logdebug('Number of sub-process created: %s' % cpucount)
        self.__logdebug('Feeding sub-process with job')
        self.__loginfo('Mapping, please wait...')

        #mapping the source to the cell using training search function
        resultlist_list = pool.map(_maptosmallest, jobpacket)
        self.__loginfo('Finish Mapping')

        mapresult_list = []
        for result in resultlist_list:
            mapresult_list.extend(result)

        self.__loginfo('Mapping post-processing')
        if self.normalize is True:
            self.__loginfo('De-normalazing the cellmap')
            self.cellmap_df = self.dfutil_func.reversescaling(self.__class__, scaller, self.paramlist)
        self.__posprocessing(title, mapresult_list)

        return self.cellmap_df



