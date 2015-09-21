from __future__ import  division
__author__ = 'mmdali'

import os
import random
import copy
import  pandas as pnd
from sklearn.preprocessing import  StandardScaler
from  sklearn.externals import joblib


class dfutil:

    def __init__(self): pass

    def feascallingdf(self , dftoscale , excludedcolumn = None ,  a =-1 , b = 1 ):

        #scalling all value in df to between b and a, where b < a

        dftouse = copy.deepcopy(dftoscale)
        if excludedcolumn:
            for col in excludedcolumn:
                del dftouse[col]

        if a >=b : raise ValueError('ERROR : a < b is expected in feascallingdf function')


        dfmin = dftouse.stack().min()
        dfmax = dftouse.stack().max()

        # formula = a + (X - Xmin )(b-a) / Xmax - Xmin

        scale = b-a
        denomitor = dfmax - dfmin

        fscaleformula = lambda x : a + (((x - dfmin)*scale)/denomitor)
        dftouse = dftouse.apply(fscaleformula)

        if excludedcolumn:
            for col in excludedcolumn:
                dftouse[col] = dftoscale[col]

        return dftouse

    def znormaliztion(self, dftonorm, excludecolumn = None, meantouse = None, stdtouse = None):

        dftouse = copy.deepcopy(dftonorm)
        if excludecolumn:
            for col in excludecolumn:
                del dftouse[col]

        if meantouse is None:
            meantouse = dftouse.stack().mean()
        if stdtouse is None:
            stdtouse = dftouse.stack().std()

        znorm = lambda x : (x - meantouse)/stdtouse
        dftouse = dftouse.apply(znorm)

        if excludecolumn:
            for col in excludecolumn:
                dftouse[col] = dftonorm[col]

        return dftouse, meantouse, stdtouse

    def stdscaler(self, dftonorm, param_list, scaler=None, projectpath=None):

        matrix = dftonorm.as_matrix(param_list)

        if scaler is None:
            scaler = StandardScaler().fit(matrix)
            matrix = scaler.transform(matrix)

            if projectpath is not None:
                scalerfilepath = os.path.join(projectpath, 'scaler')
                if not os.path.exists(scalerfilepath): os.mkdir(scalerfilepath)
                joblib.dump(scaler, os.path.join(scalerfilepath, 'scaler.pkl'))
        else:
            matrix = scaler.transform(matrix)

        matrix_df = pnd.DataFrame(matrix, columns=param_list)
        for param in param_list:
            dftonorm[param] = matrix_df[param].as_matrix()
        return dftonorm, scaler

    def reversescaling(self, dftoreverse, scaler, param_list):

        matrix = dftoreverse.as_matrix(columns=param_list)
        matrix = scaler.inverse_transform(matrix)
        matrix_df = pnd.DataFrame(matrix, columns=param_list)

        for param in param_list:
            dftoreverse[param] = matrix_df[param]

        return dftoreverse