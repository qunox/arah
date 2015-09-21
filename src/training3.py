from __future__ import  division
__author__ = 'mmdali'

import os
import time
import copy
import multiprocessing
import pandas as pnd
import numpy as np
from src import draw, dfutil
from scipy.spatial.distance import cdist

def _transformcell(packet):

    group_df = packet[0]
    minpipeend = packet[1]
    samplevec_list = packet[2]
    radius_list = packet[3]
    learconst_list = packet[4]
    irr_list = packet[5]
    paramlist = packet[6]
    plotpipe = packet[7]
    shouldplot = packet[8]
    distancetype = packet[9]
    selfpid = os.getpid()

    for irr in irr_list:
        samplevec = samplevec_list[irr]
        #find min section
        group_matrix = group_df.as_matrix(columns=paramlist)
        distanceresult = cdist(group_matrix, np.array([samplevec]), distancetype)
        mindiff = min(distanceresult)
        minindex = np.argmin(distanceresult)
        mindfindex = group_df.index.values[minindex]
        minposition = group_df.loc[mindfindex, ['x_position', 'y_position']].as_matrix()
        group_df['distance'] = distanceresult

        #giveresult to main process adn wait respond
        minpipeend.send({'job': 'min', 'mindiff': mindiff, 'position': minposition, 'pid': selfpid})

        if minpipeend.poll(600):
            letter = minpipeend.recv()

        #perturb the cellmap
        minposition, minpid = letter
        if minpid == selfpid:
            group_df.loc[mindfindex, 'pick_count'] += 1

        output_df = pnd.DataFrame(columns=paramlist)
        for index in group_df.index.values:
            distance = np.linalg.norm(group_df.loc[index, ['x_position', 'y_position']].as_matrix() - minposition)
            if distance < radius_list[irr]:
                smoothkernel = (radius_list[irr]-distance)/radius_list[irr]
                cellvec = group_df.loc[index].as_matrix(columns=paramlist)
                learnconstant = learconst_list[irr]

                #SOM Equation here
                output = cellvec + learnconstant * smoothkernel * (samplevec - cellvec)
                output_df.loc[index] = output

        for index in output_df.index.values:
            group_df.loc[index, paramlist] = output_df.loc[index, paramlist]

        #sending group_df back to main to be plotted
        if shouldplot:
            plotpipe.send({'job' :'plot', 'groupdf': group_df, 'pid':selfpid})

    #givingback the perturb df
    minpipeend.send({'job' :'finish', 'groupdf': group_df, 'pid':selfpid})


def _drawplot(inputQ):

    status = True
    waittime = 0
    draw_func = draw.draw()

    while status:
        try:
            packet = inputQ.get()
            waittime = 0
            #dftodraw, irr, drawpath, irrlist, learninglist, radiuslis, exectimelist
            if packet == 'stop': status = False
            else:
                draw_func.trainplot(packet[0], packet[1], packet[2],packet[3], packet[4], packet[5], packet[6])
        except(EOFError, IOError):
            time.sleep(0.5)
            if waittime > 18000:
                status = 'stop'
                break
            else:
                waittime += 1

class training:

    def __init__(self, samplevec_df, cellmap_df, radius_list, learningconstant_list, paramlist,project_dir, normalize=True, log=False,
                 plot=False,cpucount=-1, type = 'cossim', logcycle = False):

        self.irr_list = range(len(radius_list))
        self.samplevec_df = samplevec_df
        self.cellmap_df = cellmap_df
        self.radius_list = radius_list
        self.learningconst_list = learningconstant_list
        self.log = log
        self.draw_func = draw.draw()
        self.project_dir = project_dir

        self.normalize = normalize
        if normalize is True:
            self.dfutil_func = dfutil.dfutil()

        if type not in ['euclidean','cityblock', 'correlation',  'chebyshev', 'cosine', 'canberra', 'braycurtis', 'mahalanobis']:
            raise Exception('Unknown training type: %s' % type)
        else:
            self.traintype = type

        if cpucount == -1:
            self.cpucount = multiprocessing.cpu_count()-1
        elif cpucount == 1:
            raise Exception('Single core processing is not implemented yet')
        else:
            self.cpucount = cpucount-1

        self.param_list = paramlist

        if not logcycle is False:
            if isinstance(logcycle, int):
                self.logcylce = logcycle
            else:
                raise Exception('ERROR: logcycle should be int')

        if plot is True:
            self.plot = True
            pertubplotdir = os.path.join(self.project_dir, 'mappertub')
            if os.path.exists(pertubplotdir):
                raise IOError('ERROR: Plot directory already exist:%s' % pertubplotdir)
            os.mkdir(pertubplotdir)
            self.pertubplotdir = pertubplotdir
        else: self.plot = False

    def __loginfo(self, msg):
        if self.log:
            self.log.info(msg)

    def __logdebug(self, msg):
        if self.log:
            self.log.debug(msg)

    def __waitforrespond(self, pipelist, subprocesslist):

        result_dict = {}
        waittime = 0
        while len(result_dict.keys()) < self.cpucount:
            for mainpipe in pipelist:
                if mainpipe.poll():
                    letter = mainpipe.recv()
                    if not letter['pid'] in result_dict.keys():
                        result_dict[letter['pid']] = letter

            if waittime > 120000:
                for subp in subprocesslist:
                    subp.kill()
                raise Exception('ERROR: Timeout error in main')
            else:
                time.sleep(0.005)
                waittime += 1

        return result_dict

    def starttraining(self):

        self.__loginfo('Start training')
        self.cellmap_df['pick_count'] = [0 for i in xrange(len(self.cellmap_df))]

        #Normalizing the dataset and cellmap
        if self.normalize is False:
            self.__loginfo('WARNING CELLMAP AND SOURCE VECTOR IS NOT NORMALIZE!!')
        else:
            self.__loginfo('Normalizing the sample vec')
            self.samplevec_df , self.scaler = self.dfutil_func.stdscaler(self.samplevec_df, self.param_list)
            self.__loginfo('Normalizing the cellmap')
            self.cellmap_df, _ = self.dfutil_func.stdscaler(self.cellmap_df, self.param_list, scaler=self.scaler)

        #dividing the cellmap_df according the cpu count
        self.__logdebug('Diving the cell data frame')

        groupindex_list = np.array_split(self.cellmap_df.index.values, self.cpucount)
        groupdf_list = [self.cellmap_df.loc[groupindex] for groupindex in groupindex_list]

        #creating shared sample vector list
        self.__logdebug('Creating shared sample vec list')
        samplevec_matrix = self.samplevec_df.as_matrix(columns=self.param_list)
        manager_func = multiprocessing.Manager()
        sharedsamplevec_list = manager_func.list(samplevec_matrix)

        minmainpipes_list = []
        plotmainpipes_list = []
        packet_list = []
        self.__logdebug('Creating job packets')

        for groupdf in groupdf_list:
            minmainend, minsubend = multiprocessing.Pipe(duplex=True)
            minmainpipes_list.append(minmainend)
            plotmainend, plotsubend = multiprocessing.Pipe(duplex=True)
            plotmainpipes_list.append(plotmainend)
            packet = (groupdf, minsubend, sharedsamplevec_list, self.radius_list, self.learningconst_list, self.irr_list,
            self.param_list, plotsubend, self.plot, self.traintype)
            packet_list.append(packet)

        self.__logdebug('Creating sub-process, count: %s' % self.cpucount)
        subproces_list = []
        executiontime_list = []
        for packet in packet_list:
            subprocess = multiprocessing.Process(target=_transformcell, args=(packet,))
            subproces_list.append(subprocess)
            self.__logdebug('Starting sub-process')
            subprocess.start()
            time.sleep(0.05)

        #creating plotter subprocess
        if self.plot is True:
            plot_que = multiprocessing.Queue()
            plotsubprocess = multiprocessing.Process(target=_drawplot, args=(plot_que,))
            plotsubprocess.start()

        for irr in self.irr_list:

            #waiting to get min result from sub-process
            starttime = time.time()
            minresult_dict = self.__waitforrespond(minmainpipes_list, subproces_list)

            #giving back the min position and its pid
            mindiff_list, minpid_list, minposition_list = [], [], []
            for key in minresult_dict.keys():
                minpid_list.append(minresult_dict[key]['pid'])
                mindiff_list.append(minresult_dict[key]['mindiff'])
                minposition_list.append(minresult_dict[key]['position'])

            minindex = mindiff_list.index(min(mindiff_list))
            minpacket = [minposition_list[minindex], minpid_list[minindex]]
            for mainpipe in minmainpipes_list:
                mainpipe.send(minpacket)
            executiontime_list.append(time.time()-starttime)


            if irr % self.logcylce == 0:
                self.__loginfo('Trainning iteration: %s' % irr)
                self.__logdebug('Training Irr: %s frm: %s position: (%s,%s)' % (irr, minpid_list[minindex],
                                                                              minposition_list[minindex][0],
                                                                              minposition_list[minindex][1]))
                #plotting the the groupdf
                if self.plot:
                    plotdict = self.__waitforrespond(plotmainpipes_list, subproces_list)
                    newcellmap_df = pnd.DataFrame(columns=self.cellmap_df.columns.values)
                    for key in plotdict.keys():
                        newcellmap_df = newcellmap_df.append(plotdict[key]['groupdf'])

                    magnitude_list = [np.linalg.norm(newcellmap_df.loc[index, self.param_list]) for index in newcellmap_df.index.values]
                    newcellmap_df['magnitude'] = magnitude_list
                    self.cellmap_df = copy.deepcopy(newcellmap_df)

                    figpath = os.path.join(self.pertubplotdir, str(irr))
                    #self, dftodraw, irr, drawpath, irrlist, learninglist, radiuslis, exectimelist
                    plot_que.put([self.cellmap_df, irr, figpath, self.irr_list, self.learningconst_list,self.radius_list,
                              executiontime_list])

                    self.draw_func.trainplot(self.cellmap_df, irr, figpath, self.irr_list, self.learningconst_list,
                                             self.radius_list, executiontime_list)


        self.__logdebug('Harversting results')
        harvest_dict = self.__waitforrespond(minmainpipes_list, subproces_list)
        newcellmap_df = pnd.DataFrame(columns=self.cellmap_df.columns.values)
        for key in harvest_dict.keys():
            newcellmap_df = newcellmap_df.append(harvest_dict[key]['groupdf'])
        if self.normalize is True:
            self.__loginfo('Denormalizing the new cellmap df')
            newcellmap_df = self.dfutil_func.reversescaling(newcellmap_df, self.scaler, self.param_list)
        magnitude_list = [np.linalg.norm(newcellmap_df.loc[index, self.param_list]) for index in newcellmap_df.index.values]
        for param in self.param_list:
            self.cellmap_df[param]= newcellmap_df[param]
        self.cellmap_df['magnitude'] = magnitude_list
        self.draw_func.surfaceplt(self.cellmap_df, os.path.join(self.project_dir,'EndMagnitude.png'),'magnitude', 'magnitude')


        self.__loginfo('Training end')
        self.__logdebug('Killing all subprocess')

        for subp in subproces_list:
            subp.join()
        if self.plot is True:
            plot_que.put('stop')
            plotsubprocess.join()
        self.__logdebug('Finish killing all subprocess')

    def givecellmap(self):

        return self.cellmap_df
