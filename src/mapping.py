from __future__ import division

__author__ = 'mmdali'

import multiprocessing
import pandas as pnd
import os
import json
from random import choice
from copy import deepcopy
from numpy import array_split, linalg, dot, mean, std, exp, array
from src import draw
from scipy.spatial.distance import sqeuclidean

def _maptosmallesteuclide(jobpacket):
    cellmap_df = jobpacket[0]
    paramlist = jobpacket[1]
    mainsource_df = jobpacket[2]
    cellmaplen = len(cellmap_df)
    resultlist = []

    for sourceindex in mainsource_df.index.values:

        sourceid = mainsource_df.ix[sourceindex]['id']
        sourcelabel = mainsource_df.ix[sourceindex]['label']
        sourcevec = mainsource_df.ix[sourceindex].as_matrix(columns=paramlist).tolist()
        multisource_df = pnd.DataFrame([sourcevec for i in range(cellmaplen)], columns=paramlist)

        result_df = cellmap_df - multisource_df
        result_df = result_df[paramlist]
        func = lambda row: linalg.norm(row)
        magnitude_df = result_df.apply(func, axis=1)
        minindex = magnitude_df[magnitude_df == magnitude_df.min()].index.values[0]
        resultlist.append([minindex, sourceid, sourcelabel])

    return resultlist

def _maptosmallestdot(jobpacket):
        cellmap_df = jobpacket[0]
        paramlist = jobpacket[1]
        mainsource_df = jobpacket[2]
        resultlist = []

        for sourceindex in mainsource_df.index.values:

            sourceid = mainsource_df.ix[sourceindex]['id']
            sourcelabel = mainsource_df.ix[sourceindex]['label']
            sourcevec = mainsource_df.ix[sourceindex].as_matrix(columns=paramlist)
            squaredsourcevec = dot(sourcevec, sourcevec)
            func = lambda row: abs(squaredsourcevec - abs(dot(row.as_matrix(columns=paramlist), sourcevec)))

            result_df = cellmap_df.apply(func, axis=1)
            minindex = result_df.argmin()
            resultlist.append([minindex, sourceid, sourcelabel])

        return resultlist

def _maptosmllestcosime(jobpacket):
        cellmap_df = jobpacket[0]
        paramlist = jobpacket[1]
        mainsource_df = jobpacket[2]
        resultlist = []

        for sourceindex in mainsource_df.index.values:

            sourceid = mainsource_df.ix[sourceindex]['id']
            sourcelabel = mainsource_df.ix[sourceindex]['label']
            sourcevec = mainsource_df.ix[sourceindex].as_matrix(columns=paramlist)
            sourcemag = linalg.norm(sourcevec)
            func = lambda row: 1 - abs(dot(row.as_matrix(columns=paramlist), sourcevec)/(linalg.norm(row.as_matrix(columns=paramlist)) * sourcemag))

            result_df = cellmap_df.apply(func, axis=1)
            minindex = result_df.argmin()
            resultlist.append([minindex, sourceid, sourcelabel])

        return resultlist

def _maptosmllestrbf(jobpacket):
        cellmap_df = jobpacket[0]
        paramlist = jobpacket[1]
        mainsource_df = jobpacket[2]
        gamma = 1/len(paramlist)
        defvec = array([1.0 for i in xrange(len(paramlist))])
        resultlist = []

        for sourceindex in mainsource_df.index.values:

            sourceid = mainsource_df.ix[sourceindex]['id']
            sourcelabel = mainsource_df.ix[sourceindex]['label']
            sourcevec = mainsource_df.ix[sourceindex].as_matrix(columns=paramlist)
            func =lambda row: abs(1 - exp(gamma *sqeuclidean(row.as_matrix(columns=paramlist), sourcevec)**2))

            result_df = cellmap_df.apply(func, axis=1)
            minindex = result_df.argmin()
            resultlist.append([minindex, sourceid, sourcelabel])

        return resultlist

class mapping:
    def __init__(self, cellmap_df, normalizedsource_df, projectpath, paramlist, plot=False, log=False, probmap = False,
                 secondary=False, type='cossim'):

        self.cellmap_df = cellmap_df
        self.source_df = normalizedsource_df
        self.projectpath = projectpath

        self.paramlist = paramlist

        if secondary:
            self.secondary = True
        else:
            self.secondary = False

        if type not in ['cossim', 'euclide', 'dot', 'rbf']:
            raise Exception('Unknown mapping type: %s' % type)
        else:
            self.trainingtype = type

        if plot:
            self.plot = True
            self.draw_func = draw.draw()
            self.drawingdir = os.path.join(projectpath, 'mapcountour')
            if not secondary:
                if os.path.exists(self.drawingdir) and not probmap:
                    raise IOError('ERROR: Drawing path already exist:%s' % self.drawingdir)
                elif not probmap:
                    os.mkdir(self.drawingdir)

        if log:
            self.log = log


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
            sourcecount_list.append(len(self.cellmap_df.ix[index][name +'source_id']))
            signalcount_list.append(self.cellmap_df.ix[index][name +'source_label'].count(1.0))
            noisesecount_list.append(self.cellmap_df.ix[index][name +'source_label'].count(0.0))

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

        if self.plot:
            self.__loginfo('Making contour plots')
            sourcecount_plt = os.path.join(self.drawingdir, name +'source_count_map.png')
            self.draw_func.contourfplot(self.cellmap_df, sourcecount_plt, name +'source_count',
                                       title=name +'Event count distribution')

            self.__loginfo('Making surface plots')
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

    #deprecated, and questionble, do not understand why have to creatre a mapresult_dict
    def __mapsource(self,  cpucount=None):

        if cpucount is None:
            cpucount = multiprocessing.cpu_count() - 1
        self.__logdebug('Dividing the source data frame')

        #deviding the source by the number of process spawned
        minisourcedflist = array_split(self.source_df, cpucount)

        #creating job packets
        _multiplecellmaplist = [deepcopy(self.cellmap_df) for i in range(cpucount)]
        _multiparamlist = [self.paramlist for i in range(cpucount)]
        jobpacket = zip(_multiplecellmaplist, _multiparamlist, minisourcedflist)

        self.__logdebug('Creating multiprocessing pool')
        pool = multiprocessing.Pool(cpucount)
        self.__logdebug('Number of sub-process created: %s' % cpucount)
        self.__logdebug('Feeding sub-process with job')
        self.__loginfo('Mapping, please wait...')

        #mapping the source to the cell using training search function
        if self.trainingtype == 'euclide':
            resultlist_list = pool.map(_maptosmallesteuclide, jobpacket)
        elif self.trainingtype == 'cossim':
            resultlist_list = pool.map(_maptosmllestcosime, jobpacket)
        elif self.trainingtype == 'dot':
            resultlist_list = pool.map(_maptosmallestdot, jobpacket)

        pool.close()
        pool.join()
        self.__loginfo('Finish Mapping')
        mapresult_list = []
        for result in resultlist_list:
            mapresult_list.extend(result)

        self.__loginfo('Mapping pre-processing')

        if not self.secondary:
            self.__posprocessing('prime_', mapresult_list)
        if self.secondary:

            self.__posprocessing('secondary_', mapresult_list)

            self.cellmap_df['count_diff'] = self.cellmap_df['secondary_source_count'] - self.cellmap_df['prime_source_count']

            plotpath = os.path.join(self.drawingdir, 'source_count_diff_map.png')
            self.draw_func.contourfplot(self.cellmap_df, plotpath, 'count_diff', title='Source count diff')
            plotpath = os.path.join(self.drawingdir, 'source_count_diff_srf.png')
            self.draw_func.surfaceplt(self.cellmap_df, plotpath, 'count_diff', title='Source count diff')

        def fillmaprresult_dic(mapresultdic, name):

            mapresultdic[name] = {'1std': {'idlist': []}, '2std': {'idlist': []}, '3std': {'idlist': []}}
            for index in self.cellmap_df.index.values:
                excessratio = self.cellmap_df.loc[index, name+'_excess_ratio']
                if excessratio > 3:
                    mapresultdic[name]['3std']['idlist'].append(self.cellmap_df.loc[index, 'id'])
                elif excessratio >2 and excessratio < 3:
                    mapresultdic[name]['2std']['idlist'].append(self.cellmap_df.loc[index, 'id'])
                elif excessratio >1 and excessratio < 2:
                    mapresultdic[name]['1std']['idlist'].append(self.cellmap_df.loc[index, 'id'])

            for key in mapresultdic[name].keys():
                mapresultdic[name][key]['count'] = len(mapresultdic[name][key]['idlist'])

            if name == 'diff':
                mapresultdic[name]['n1std'] = {'idlist': []}
                mapresultdic[name]['n2std'] = {'idlist': []}
                mapresultdic[name]['n3std'] = {'idlist': []}

                for index in self.cellmap_df.index.values:
                    excessratio = self.cellmap_df.loc[index, name+'_excess_ratio']
                    if excessratio < -3:
                        mapresultdic[name]['n3std']['idlist'].append(self.cellmap_df.loc[index, 'id'])
                    elif excessratio < -2 and excessratio > -3 :
                        mapresultdic[name]['n2std']['idlist'].append(self.cellmap_df.loc[index, 'id'])
                    elif excessratio > -2 and excessratio < -1:
                        mapresultdic[name]['n1std']['idlist'].append(self.cellmap_df.loc[index, 'id'])

                for key in mapresultdic[name].keys():
                    mapresultdic[name][key]['count'] = len(mapresultdic[name][key]['idlist'])

        mapresult_dict = {}
        if not self.secondary:
            fillmaprresult_dic(mapresult_dict, 'prime')
        if self.secondary:
            fillmaprresult_dic(mapresult_dict, 'secondary')

            meandiff = mean(self.cellmap_df['count_diff'])
            stddiff = std(self.cellmap_df['count_diff'])
            diffexcess_list = []
            for index in self.cellmap_df.index.values:
                diff = self.cellmap_df.loc[index, 'count_diff']
                diffexcess_list.append((diff - meandiff)/(stddiff))
            self.cellmap_df['diff_excess_ratio'] = diffexcess_list

            fillmaprresult_dic(mapresult_dict, 'diff')

        self.__loginfo('Saving mapping dict')
        mapdictpath = os.path.join(self.projectpath, 'mapresult.json')
        if os.path.exists(mapdictpath):
            loadfile = open(mapdictpath, 'r')
            oldcit = json.load(loadfile)
            for key in oldcit.keys():
                mapresult_dict[key] = oldcit[key]
            loadfile.close()

        dumpfile = open(mapdictpath, 'w')
        json.dump(mapresult_dict,dumpfile)
        dumpfile.close()

        return self.cellmap_df

    def mapsource(self, cpucount=None):

        if cpucount is None:
            cpucount = multiprocessing.cpu_count() - 1
        self.__logdebug('Dividing the source data frame')

        #deviding the source by the number of process spawned
        minisourcedflist = array_split(self.source_df, cpucount)

        #creating job packets
        _multiplecellmaplist = [deepcopy(self.cellmap_df) for i in range(cpucount)]
        _multiparamlist = [self.paramlist for i in range(cpucount)]
        jobpacket = zip(_multiplecellmaplist, _multiparamlist, minisourcedflist)

        self.__logdebug('Creating multiprocessing pool')
        pool = multiprocessing.Pool(cpucount)
        self.__logdebug('Number of sub-process created: %s' % cpucount)
        self.__logdebug('Feeding sub-process with job')
        self.__loginfo('Mapping, please wait...')

        #mapping the source to the cell using training search function
        if self.trainingtype == 'euclide':
            resultlist_list = pool.map(_maptosmallesteuclide, jobpacket)
        elif self.trainingtype == 'cossim':
            resultlist_list = pool.map(_maptosmllestcosime, jobpacket)
        elif self.trainingtype == 'dot':
            resultlist_list = pool.map(_maptosmallestdot, jobpacket)
        elif self.trainingtype == 'rbf':
            resultlist_list = pool.map(_maptosmllestrbf, jobpacket)

        pool.close()
        pool.join()
        self.__loginfo('Finish Mapping')
        mapresult_list = []
        for result in resultlist_list:
            mapresult_list.extend(result)

        self.__loginfo('Mapping pre-processing')

        if not self.secondary:
            self.__posprocessing('prime_', mapresult_list)

        if self.secondary:
            self.__posprocessing('secondary_', mapresult_list)
            self.cellmap_df['count_diff'] = self.cellmap_df['secondary_source_count'] - self.cellmap_df['prime_source_count']

            meandiff = mean(self.cellmap_df['count_diff'])
            stddiff = std(self.cellmap_df['count_diff'])
            diffexcess_list = []
            for index in self.cellmap_df.index.values:
                diff = self.cellmap_df.loc[index, 'count_diff']
                diffexcess_list.append((diff - meandiff)/(stddiff))
            self.cellmap_df['count_diff_excess_ratio'] = diffexcess_list

            plotpath = os.path.join(self.drawingdir, 'source_count_diff_map.png')
            self.draw_func.contourfplot(self.cellmap_df, plotpath, 'count_diff', title='Source count diff')
            plotpath = os.path.join(self.drawingdir, 'source_count_diff_srf.png')
            self.draw_func.surfaceplt(self.cellmap_df, plotpath, 'count_diff', title='Source count diff')



        return self.cellmap_df


