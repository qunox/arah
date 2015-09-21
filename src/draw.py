from __future__ import  division
__author__ = 'mmdali'

import copy
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

class draw:

    def __init__(self): pass

    def learningconstent(self, packet):

            irr_list = packet['irr_list']
            learningcontant_list = packet['learningcontant_list']
            traininglenght = packet['traininglenght']
            k = packet['k']
            L = packet['L']
            midpoint = packet['midpoint']
            projectpath = packet['projectpath']

            fig = plt.figure()
            fig.suptitle('Learning Constant Decay', fontsize=14, fontweight='bold')
            ax = fig.add_subplot(111)
            ax.plot(irr_list, learningcontant_list)
            ax.grid()
            ax.set_title('Learning Iteration:%s k:%s L:%s x0:%s' % (traininglenght,k ,L,midpoint))

            plt.savefig(projectpath)
            plt.close()

    def radiusdecay(self, packet):

            irr_list = packet['irr_list']
            radiuslist = packet['radiuslist']
            traininglenght = packet['traininglenght']
            k = packet['k']
            L = packet['L']
            midpoint = packet['midpoint']
            projectpath = packet['projectpath']

            plt.plot(irr_list, radiuslist)
            plt.grid()
            plt.title('Radius Decay')
            plt.suptitle('Iteration:%s k:%s L:%s x0:%s' % (traininglenght,k ,L,midpoint))
            plt.savefig(projectpath)
            plt.close()

    def givemeshgridd(self, dftodraw,columntodraw):

        if not 'x_position' in dftodraw.columns.values or not 'y_position' in dftodraw.columns.values:
            raise ValueError('ERROR: Failed to draw, data frame given does have "x_positon" or "y_position" column')

        xmax = int(dftodraw['x_position'].max())
        ymax = int(dftodraw['y_position'].max())
        xmin = int(dftodraw['x_position'].min())
        ymin = int(dftodraw['y_position'].min())

        x_list = np.array(range(xmin, xmax+1))
        y_list = np.array(range(ymin, ymax+1))

        xx, yy = np.meshgrid(x_list, y_list)
        zz = [[None for y in y_list] for x in x_list]

        for index in dftodraw.index:

            xpositon = int(dftodraw.ix[index]['x_position'])
            yposition = int(dftodraw.ix[index]['y_position'])

            value = dftodraw.ix[index][columntodraw]

            zz[xpositon][yposition] = value
        zz = np.array(zz)

        return xx,yy,zz

    def surfaceplt(self, dftodraw, drawpath, columntodraw, title = False):


        xx, yy, zz = self.givemeshgridd(dftodraw,columntodraw)

        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False, alpha=0.5)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        if title:
            plt.title(title)
        plt.savefig(drawpath)
        plt.close()

    def doublecontourplot(self, dftodraw, drawpath, linecolumntodraw, clrcolumntodraw, title = False, transparent=False):

        lxx, lyy, lzz = self.givemeshgridd(dftodraw, linecolumntodraw)
        cxx, cyy, czz = self.givemeshgridd(dftodraw, clrcolumntodraw)

        plt.contour(lxx, lyy, lzz,15, colors='k')
        CS = plt.contourf(cxx, cyy, czz, 15, cmap=plt.cm.coolwarm)
        plt.colorbar()

        if title: plt.title(title)
        if transparent is True:
            plt.savefig(drawpath, transparent=True)
        else:
            plt.savefig(drawpath)
        plt.close()

    def contourfplot(self, dftodraw, drawpath, columntodraw, title = False, transparent=False, pocodot=False):

        xx, yy, zz = self.givemeshgridd(dftodraw, columntodraw)

        CS = plt.contour(xx, yy, zz, 15, linewidths=0.5)
        plt.clabel(CS, inline=1, fontsize=10)
        #CS = plt.contourf(xx, yy, zz, 15, cmap=plt.cm.coolwarm,vmax=abs(zz).max(), vmin=-abs(zz).max())
        CS = plt.contourf(xx, yy, zz, 15, cmap=plt.cm.coolwarm)
        plt.colorbar()
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        if pocodot is True:
            plt.scatter(dftodraw['x_position'], dftodraw['y_position'])

        if title: plt.title(title)
        if transparent is True:
            plt.savefig(drawpath, transparent=True)
        else:
            plt.savefig(drawpath)
        plt.close()

    def markedcontourplot(self, dftodraw, drawpath, columntodraw, xlist, ylist, title = False, transparent=False ):

        xx, yy, zz = self.givemeshgridd(dftodraw, columntodraw)

        CS = plt.contour(xx, yy, zz, 15, linewidths=0.5)
        plt.clabel(CS, inline=1, fontsize=10)
        CS = plt.contourf(xx, yy, zz, 15, cmap=plt.cm.coolwarm,vmax=abs(zz).max(), vmin=-abs(zz).max())
        plt.colorbar()

        plt.scatter(xlist,ylist, color='black', marker='x')

        if title: plt.title(title)
        if transparent is True:
            plt.savefig(drawpath, transparent=True)
        else:
            plt.savefig(drawpath)
        plt.close()


    def trainplot(self, dftodraw, irr, drawpath, irrlist, learninglist, radiuslis, executiontime_list, columntodraw='magnitude'):

        fig = plt.figure(figsize=(15,10), dpi=100)
        fig.suptitle('Iteration: %s' % irr, fontsize=12, fontweight='bold')

        ax1 = plt.subplot2grid((4,4),(0,0), colspan=2, rowspan=3, projection='3d')
        xx, yy, zz = self.givemeshgridd(dftodraw, columntodraw)
        surf = ax1.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=True, alpha=0.5)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax1.set_title(columntodraw +' Topology')

        ax4 = plt.subplot2grid((4,4),(0,2), colspan=2, rowspan=3, projection='3d')
        pxx, pyy, pzz = self.givemeshgridd(dftodraw, 'pick_count')
        surf2 = ax4.plot_surface(pxx, pyy, pzz, rstride=1, cstride=1, cmap=cm.rainbow,linewidth=0, antialiased=True, alpha=0.5)
        ax4.set_title('Pick count')

        ax2 = plt.subplot2grid((4,4),(3,0))
        ax2.plot(irrlist, learninglist)
        ax2.plot([irr, irr], [0, 1], color='red')
        ax2.grid()
        ax2.set_title('Learning Constant')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Learn-factor')

        ax3 = plt.subplot2grid((4,4),(3,1))
        ax3.plot(irrlist, radiuslis)
        ax3.plot([irr, irr],[0, max(radiuslis)], color='red')
        ax3.grid()
        ax3.set_title('Radius Decay')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Radius')

        picklist = dftodraw['pick_count'].as_matrix().tolist()
        ax5 = plt.subplot2grid((4,4),(3,2))
        ax5.hist(picklist, 100)
        mean, std = np.mean(picklist), np.std(picklist)
        ax5.set_title('mean=%.3g std=%.3g' % (mean, std))

        if irr != 0:
            irrdone = [irrlist[index] for index in range(len(executiontime_list))]
            ax6 = plt.subplot2grid((4,4),(3,3))
            ax6.plot(irrdone[1:],executiontime_list[1:])
            ax6.set_title('Exe Time')
            ax6.set_xlabel('Iteration')
            ax6.set_ylabel('Time(s)')

        plt.savefig(drawpath)
        plt.close()