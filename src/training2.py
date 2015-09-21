from __future__ import  division
__author__ = 'mmdali'

import os
import time
import copy
import multiprocessing
import pandas as pnd
import numpy as np
from src import draw
from scipy.spatial.distance import sqeuclidean

def _transformcelleuclide(packet):

    group_df = packet[0]
    minpipeend = packet[1]
    samplevec_list = packet[2]
    radius_list = packet[3]
    learconst_list = packet[4]
    irr_list = packet[5]
    paramlist = packet[6]
    plotpipe = packet[7]
    shouldplot = packet[8]
    selfpid = os.getpid()

    for irr in irr_list:
        samplevec = samplevec_list[irr]

        #find min section
        func = lambda row: np.linalg.norm(samplevec - row.as_matrix(columns=paramlist))
        euclide_df = group_df.apply(func, axis=1)
        mindiff = euclide_df.min(skipna=True)
        minindex = euclide_df.argmin(skipna=True)
        minposition = group_df.loc[minindex, ['x_position', 'y_position']].as_matrix()

        #giveresult to main process adn wait respond
        minpipeend.send({'job': 'min', 'mindiff': mindiff, 'position': minposition, 'pid': selfpid})
        if minpipeend.poll(600):
            letter = minpipeend.recv()

        #perturb the cellmap
        minposition, minpid = letter
        if minpid == selfpid:
            group_df.loc[minindex, 'pick_count'] += 1

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

def _transformcelldot(packet):

    group_df = packet[0]
    minpipeend = packet[1]
    samplevec_list = packet[2]
    radius_list = packet[3]
    learconst_list = packet[4]
    irr_list = packet[5]
    paramlist = packet[6]
    plotpipe = packet[7]
    shouldplot = packet[8]
    selfpid = os.getpid()

    for irr in irr_list:
        samplevec = samplevec_list[irr]

        #find min section
        squaredvec = np.dot(samplevec, samplevec)
        func = lambda row: abs(squaredvec - abs(np.dot(samplevec, row.as_matrix(columns=paramlist))))
        dot_df = group_df.apply(func, axis=1)
        mindiff = dot_df.min(skipna=True)
        minindex = dot_df.argmin(skipna=True)
        minposition = group_df.loc[minindex, ['x_position', 'y_position']].as_matrix()

        #giveresult to main process adn wait respond
        minpipeend.send({'job': 'min', 'mindiff': mindiff, 'position': minposition, 'pid': selfpid})
        if minpipeend.poll(600):
            letter = minpipeend.recv()

        #perturb the cellmap
        minposition, minpid = letter
        if minpid == selfpid:
            group_df.loc[minindex, 'pick_count'] += 1

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

def _transformcellcosine(packet):

    group_df = packet[0]
    minpipeend = packet[1]
    samplevec_list = packet[2]
    radius_list = packet[3]
    learconst_list = packet[4]
    irr_list = packet[5]
    paramlist = packet[6]
    plotpipe = packet[7]
    shouldplot = packet[8]
    selfpid = os.getpid()

    for irr in irr_list:
        samplevec = samplevec_list[irr]
        samplevecmag = np.linalg.norm(samplevec)
        #find min section
        func = lambda row: 1 -abs((np.dot(samplevec, row.as_matrix(columns=paramlist)))/(np.linalg.norm(row.as_matrix(columns=paramlist) * samplevecmag)))
        cosine_df = group_df.apply(func, axis=1)
        mindiff = cosine_df.min(skipna=True)
        minindex = cosine_df.argmin(skipna=True)
        minposition = group_df.loc[minindex, ['x_position', 'y_position']].as_matrix()

        #giveresult to main process adn wait respond
        minpipeend.send({'job': 'min', 'mindiff': mindiff, 'position': minposition, 'pid': selfpid})
        if minpipeend.poll(600):
            letter = minpipeend.recv()

        #perturb the cellmap
        minposition, minpid = letter
        if minpid == selfpid:
            group_df.loc[minindex, 'pick_count'] += 1

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

def _transformcellrbf(packet):

    group_df = packet[0]
    minpipeend = packet[1]
    samplevec_list = packet[2]
    radius_list = packet[3]
    learconst_list = packet[4]
    irr_list = packet[5]
    paramlist = packet[6]
    plotpipe = packet[7]
    shouldplot = packet[8]
    selfpid = os.getpid()
    gamma = 1/len(paramlist)

    for irr in irr_list:
        samplevec = samplevec_list[irr]
        try:
            #find min section
            func = lambda row: abs(1 - np.exp(gamma *sqeuclidean(row.as_matrix(columns=paramlist), samplevec)**2))
            rbf_df = group_df.apply(func, axis=1)
            mindiff = rbf_df.min(skipna=True)
            minindex = rbf_df.argmin(skipna=True)
            minposition = group_df.loc[minindex, ['x_position', 'y_position']].as_matrix()
        except Exception, msg:
            print 'Msg: ', msg
            print 'samplevec', samplevec
            print 'irr: ', irr
            print 'rfb_df: ', rbf_df
            print 'mindiff', mindiff
            print 'argmin: ', rbf_df.argmin(skipna=True)

        #giveresult to main process adn wait respond
        minpipeend.send({'job': 'min', 'mindiff': mindiff, 'position': minposition, 'pid': selfpid})
        if minpipeend.poll(600):
            letter = minpipeend.recv()

        #perturb the cellmap
        minposition, minpid = letter
        if minpid == selfpid:
            group_df.loc[minindex, 'pick_count'] += 1

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


def _drawplot(inputQ):

    status = True
    waittime = 0
    draw_func = draw.draw()

    while status:
        try:
            packet = inputQ.get()
            waittime = 0
            #dftodraw, irr, drawpath, irrlist, learninglist, radiuslis, exectimelist
            draw_func.trainplot(packet[0], packet[1], packet[2],packet[3], packet[4], packet[5], packet[6])
        except(EOFError, IOError):
            time.sleep(0.5)
            if waittime > 18000:
                status = 'stop'
                break
            else:
                waittime += 1

class training:

    def __init__(self, samplevec_df, cellmap_df, radius_list, learningconstant_list, paramlist,  log=False,
                 plot=False, project_dir=False, cpucount=-1, type = 'cossim', logcycle = False):

        self.irr_list = range(len(radius_list))
        self.samplevec_df = samplevec_df
        self.cellmap_df = cellmap_df
        self.radius_list = radius_list
        self.learningconst_list = learningconstant_list
        self.log = log

        if type not in ['cossim', 'dot', 'euclide', 'rbf']:
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

        if project_dir:
            self.project_dir = project_dir

        if plot is True:
            self.plot = True
            self.draw_func = draw.draw()
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

            if waittime > 60000:
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
            packet = []
            minmainend, minsubend = multiprocessing.Pipe(duplex=True)
            minmainpipes_list.append(minmainend)
            plotmainend, plotsubend = multiprocessing.Pipe(duplex=True)
            plotmainpipes_list.append(plotmainend)

            packet.append(groupdf)
            packet.append(minsubend)
            packet.append(sharedsamplevec_list)
            packet.append(self.radius_list)
            packet.append(self.learningconst_list)
            packet.append(self.irr_list)
            packet.append(self.param_list)
            packet.append(plotsubend)
            packet.append(self.plot)
            packet_list.append(packet)

        self.__logdebug('Creating sub-process, count: %s' % self.cpucount)
        subproces_list = []
        executiontime_list = []
        for packet in packet_list:
            if self.traintype == 'euclide':
                subprocess = multiprocessing.Process(target=_transformcelleuclide, args=(packet,))
            elif self.traintype == 'cossim':
                subprocess = multiprocessing.Process(target=_transformcellcosine, args=(packet,))
            elif self.traintype == 'dot':
                subprocess = multiprocessing.Process(target=_transformcelldot, args=(packet,))
            elif self.traintype == 'rbf':
                subprocess = multiprocessing.Process(target=_transformcellrbf, args=(packet,))
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

        self.__loginfo('Training end')

    def givecellmap(self):

        return self.cellmap_df
