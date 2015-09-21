from __future__ import  division
__author__ = 'mmdali'

import numpy as np
import os
import random
import multiprocessing
import copy
from sklearn.cluster import DBSCAN
from sklearn.svm import SVC
from matplotlib import colors
from matplotlib import  pyplot as plt
from util import cellmaputil
from scipy import stats


def _shufling(packet):

    cellindex_list = packet[0]
    realsamplesize = packet[1]
    cellprob_dict = packet[2]

    random.shuffle(cellindex_list)
    sampleindex = random.sample(cellindex_list, realsamplesize)
    for index in sampleindex:
        cellprob_dict[index]['pick'] += 1
    for key in cellprob_dict.keys():
        cellprob_dict[key]['picklist'].append(cellprob_dict[key]['pick'])
        cellprob_dict[key]['pick'] = 0

    return cellprob_dict

class mapanalysis:

    def __init__(self, cellmap_df, paramlist, log = False):

        self.cellmap_df = cellmap_df
        self.log = log
        self.cellutil = cellmaputil.cellmaputil()
        self.prefix_list = []
        self.paramlist = paramlist
        for col in self.cellmap_df.columns.values:
            if 'source_id' in col: self.prefix_list.append(col.replace('source_id', ''))

    def __loginfo(self, msg):
        if self.log: self.log.info(msg)

    def __logdebug(self, msg):
        if self.log: self.log.debug(msg)

    def probmap(self, samplesize = 0.3, iteration = 100):

        self.__loginfo('Begin Probability mapping')
        self.__logdebug('Starting pool processes')
        pool = multiprocessing.Pool()

        #analyzing for earch prefix, ie prime and secondary
        for prefix in self.prefix_list:
            self.__logdebug('Analysing prefix: %s' % prefix)

            sourceid_list = []
            cellindex_list = []

            for index in self.cellmap_df.index.values:
                sourceids = self.cellmap_df.ix[index][prefix+'source_id']
                if len(sourceids) > 0:
                    for id in sourceids:
                        if not id in sourceid_list:
                            sourceid_list.append(id)
                            cellindex_list.append(index)

            #creating a main probability dick
            realsamplesize = int(samplesize * len(sourceid_list))
            cellprob_dict = {}
            for index in self.cellmap_df.index.values:
                cellprob_dict[index] = {'pick': 0, 'picklist': []}

            self.__logdebug('Creating job packet')
            packets_list = [[cellindex_list, realsamplesize, copy.deepcopy(cellprob_dict)] for irr in xrange(iteration)]
            mapresult_list = pool.map(_shufling, packets_list)

            for mapcellprob_dict in mapresult_list:
                for index in self.cellmap_df.index.values:
                    cellprob_dict[index]['picklist'].extend(mapcellprob_dict[index]['picklist'])

            for key in cellprob_dict.keys():
                pick_list = cellprob_dict[key]['picklist']
                prob_list = [value/realsamplesize for value in pick_list]
                cellprob_dict[key]['problist'] = prob_list

                #calculating the expected value
                value_list = []
                for prob in prob_list:
                    if prob not in value_list: value_list.append(prob)
                valuecount = len(value_list)
                valueprob_list = [prob_list.count(val)/valuecount for val in value_list]
                expact = stats.rv_discrete(name='expact', values=(value_list, valueprob_list))

                cellprob_dict[key]['expectation'] = expact.mean()
                cellprob_dict[key]['std'] = expact.std()
                cellprob_dict[key]['1s'] = cellprob_dict[key]['expectation'] + cellprob_dict[key]['std']
                cellprob_dict[key]['2s'] = cellprob_dict[key]['expectation'] + 2 * cellprob_dict[key]['std']
                cellprob_dict[key]['3s'] = cellprob_dict[key]['expectation'] + 3 * cellprob_dict[key]['std']

            completeprob_list = []
            completeexpactation_list = []
            completestd_list = []
            complete1s_list = []
            complete2s_list = []
            complete3s_list = []

            for index in self.cellmap_df.index.values:
                if index in cellprob_dict.keys():
                    completeprob_list.append(cellprob_dict[index]['problist'])
                    completeexpactation_list.append(cellprob_dict[index]['expectation'])
                    completestd_list.append(cellprob_dict[index]['std'])
                    complete1s_list.append(cellprob_dict[index]['1s'])
                    complete2s_list.append(cellprob_dict[index]['2s'])
                    complete3s_list.append(cellprob_dict[index]['3s'])
                else:
                    completeprob_list.append([])
                    completeexpactation_list.append(0.0)
                    completestd_list.append(0.0)
                    complete1s_list.append(0.0)
                    complete2s_list.append(0.0)
                    complete3s_list.append(0.0)

            self.cellmap_df[prefix + 'prob_list'] = completeprob_list
            self.cellmap_df[prefix + 'expectation'] = completeexpactation_list
            self.cellmap_df[prefix + 'std'] = completestd_list
            self.cellmap_df[prefix + '1sigma'] = complete1s_list
            self.cellmap_df[prefix + '2sigma'] = complete2s_list
            self.cellmap_df[prefix + '3sigma'] = complete3s_list

            self.__logdebug('Finish for prefix: %s' % prefix)

        self.__logdebug('Cloosing pool process')
        pool.close()
        pool.join()
        return self.cellmap_df

    def regroup(self, maxdistance, minsize, algo = 'auto'):

        self.__loginfo('Regrouping')
        dbsfit = DBSCAN(eps=maxdistance, min_samples=minsize, algorithm=algo).fit(self.primarylist)
        dbsresult = dbsfit.fit_predict(self.primarylist)
        grouplist = []
        for grouplabel in dbsresult:
            if not grouplabel in grouplist: grouplist.append(grouplabel)
        self.__loginfo('Group label count: %s' % len(grouplist))

    #deprecated, used DBSCAN withc hard to paramatize
    def densmap(self, maxdistance = 1, minsize = 100, algo = 'auto', plot=False, projectpath = None, interactive = False):

        if plot is True and projectpath is None:
            raise Exception('Plot directory is not given')
        else:
            plot_dir = os.path.join(projectpath, 'dbsplot')
            if os.path.exists(plot_dir):
                raise Exception('Directory already exist %s' % plot_dir)
            else:
                os.mkdir(plot_dir)

        self.__loginfo('Begin DBS clustering')

        for prefix in self.prefix_list:
            xlist, ylist, indexlist = [], [], []

            self.__logdebug('Creating X-Y scatter list for source for %s' % prefix)
            for index in self.cellmap_df.index.values:
                sourcecount = self.cellmap_df.ix[index][prefix+'source_count']
                if sourcecount > 0:
                    xexten = [self.cellmap_df.ix[index]['x_position'] for i in xrange(sourcecount)]
                    yexten = [self.cellmap_df.ix[index]['y_position'] for i in xrange(sourcecount)]
                    xlist.extend(xexten)
                    ylist.extend(yexten)
                    indexlist.append(index)

            primarylist = np.array([[x, y] for x, y in zip(xlist, ylist)])

            self.__logdebug('Begin DBS fitting for %s' % prefix)
            dbsfit = DBSCAN(eps=maxdistance, min_samples=minsize, algorithm=algo).fit(primarylist)
            dbsresult = dbsfit.fit_predict(primarylist)

            cellgroup_list = []
            celllocation_list = []
            for location, group in zip(primarylist, dbsresult):
                if group != -1:
                    cellgroup_list.append(group)
                    celllocation_list.append(location)
            celllocation_list = np.array(celllocation_list)
            cellgroup_list = np.array(cellgroup_list)
            self.__logdebug('Beging SVC fitting for: %s' % prefix)
            self.primarylist = primarylist
            try:
                svcfit = SVC(kernel = 'rbf', C=1).fit(celllocation_list, cellgroup_list)
            except ValueError, msg:
                self.__loginfo('Initial dbs group failed')
                if interactive is True:
                    import pdb
                    pdb.set_trace()
                    #====WELCOME TO DBS INTERACTIVE====
                else:
                    raise ValueError(msg)

            alllocation = [[x,y] for x,y in zip(self.cellmap_df['x_position'], self.cellmap_df['y_position'])]
            alllocation = np.array(alllocation)
            svcresult = svcfit.predict(alllocation)
            self.cellmap_df[prefix + 'dbs_group'] = svcresult
            self.__logdebug('Finish DBS fitting for: %s ' % prefix)

            if plot:
                self.__loginfo('Ploting the DBS result for: %s ' % prefix)
                plotting_dict = {}
                colorindex = 0
                color_list = colors.cnames.keys()
                allx_list = []
                ally_list = []
                for group, location in zip(svcresult, alllocation):
                    if not group in plotting_dict.keys():
                        plotting_dict[group] = {'x': [], 'y': [], 'color':color_list[colorindex]}
                        colorindex += 1
                    plotting_dict[group]['x'].append(location[0])
                    plotting_dict[group]['y'].append(location[1])
                    allx_list.append(location[0])
                    ally_list.append(location[1])

                if len(plotting_dict.keys()) == 1: self.__loginfo('WARNING: Only one group is created in DBS mapping')

                fig = plt.figure(figsize=(8,8), dpi=100,)
                ax = fig.add_subplot(111)
                for group in plotting_dict.keys():
                    ax.scatter(plotting_dict[group]['x'], plotting_dict[group]['y'],
                               color=plotting_dict[group]['color'], label=group)

                hist = ax.hist2d(xlist, ylist, self.cellmap_df['x_position'].max(), alpha= 0.7)
                fig.colorbar(hist[3], ax=ax)

                plt.title(prefix+'Density Based Clustering')
                plt.legend()
                plt.savefig(os.path.join(plot_dir, prefix + 'dbsresult.png'))
                plt.close()

        return self.cellmap_df

    def densmap2(self,projectpath, coltouse='secondary', minstd=2, plot=False):

        self.__loginfo('Beging K-Means Clustering')

        cellid_list = []
        coltouse = coltouse+'_excess_ratio'
        for index in self.cellmap_df.index.values:
            if self.cellmap_df.loc[index, coltouse] >= minstd:
                cellid_list.append(self.cellmap_df.loc[index, 'id'])

        #making a centroid df matrix
        centroid_df = self.cellmap_df.loc[self.cellmap_df['id'].isin(cellid_list)]
        centroidparam_list = ['x_position', 'y_position']
        centroidparam_list.extend(self.paramlist)
        centroid_matrix = centroid_df.as_matrix(columns=centroidparam_list)
        centroidnum = len(centroid_matrix)

        #preparing the data
        cellmap_matrix = self.cellmap_df.as_matrix(columns=centroidparam_list)

        #start K-Means clustering
        from sklearn.cluster import KMeans

        self.__loginfo('Staring K-Means Clustering')
        kmeans_func = KMeans(centroidnum, init=centroid_matrix, max_iter=1000, n_jobs= -2, n_init=1)
        self.__logdebug('Staring K-Means Fitting')
        kmeansfit = kmeans_func.fit(cellmap_matrix)
        kmeanresult_list = kmeansfit.predict(cellmap_matrix)

        #attaching the result to cellmap
        self.cellmap_df['kmeasn_group'] = kmeanresult_list
        #plotting if true

    def __densmap2(self,projectpath, coltouse='secondary', minstd=2, plot=False):

        self.__loginfo('Beging K-Means Clustering')
        '''
        self.__logdebug('Loading mapping dict')

        mapdictpath = os.path.join(projectpath,'mapresult.json')
        dictfile = open(mapdictpath, 'r')
        mapping_dict = json.load(dictfile)
        dictfile.close()

        #taking the position and number of initial k-point from mapping dict
        maptypetouse_dict = mapping_dict[coltouse]
        if minstd == 3:
            stdtouse_list = ['3std', 'n3std']
        elif minstd == 2:
            stdtouse_list = ['3std', 'n3std', '2std', 'n2std']
        elif minstd == 1:
            stdtouse_list = ['3std', 'n3std', '2std', 'n2std', '1std', 'n1std']
        else:
            raise Exception('Wrong min std level')

        cellid_list = []
        for key in stdtouse_list:
            if key in maptypetouse_dict.keys():
                cellid_list.extend(maptypetouse_dict[key]['idlist'])
        '''

        cellid_list = []
        coltouse = coltouse+'_excess_ratio'
        for index in self.cellmap_df.index.values:
            if self.cellmap_df.loc[index, coltouse] >= minstd:
                cellid_list.append(self.cellmap_df.loc[index, 'id'])

        #making a centroid df matrix
        centroid_df = self.cellmap_df.loc[self.cellmap_df['id'].isin(cellid_list)]
        centroidparam_list = ['x_position', 'y_position']
        centroidparam_list.extend(self.paramlist)
        centroid_matrix = centroid_df.as_matrix(columns=centroidparam_list)
        centroidnum = len(centroid_matrix)

        #preparing the data
        cellmap_matrix = self.cellmap_df.as_matrix(columns=centroidparam_list)

        #start K-Means clustering
        from sklearn.cluster import KMeans

        self.__loginfo('Staring K-Means Clustering')
        kmeans_func = KMeans(centroidnum, init=centroid_matrix, max_iter=1000, n_jobs= -2, n_init=1)
        self.__logdebug('Staring K-Means Fitting')
        kmeansfit = kmeans_func.fit(cellmap_matrix)
        kmeanresult_list = kmeansfit.predict(cellmap_matrix)

        #attaching the result to cellmap
        self.cellmap_df['kmeasn_group'] = kmeanresult_list
        #plotting if true