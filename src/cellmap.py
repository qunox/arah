from __future__ import division
__author__ = 'mmdali'

import random

import pandas as pnd
from uuid import  uuid4

class cellmap:

    def __init__(self): pass

    def givecellmap(self, mapwidth , mapheight):

        self.mapwidt = mapwidth
        self.mapheight = mapheight

        cellmap_df = pnd.DataFrame()
        cellx_list = []
        celly_list = []
        celluuid_list = []

        for x in range(mapwidth):
            for y in range(mapheight):
                cellx_list.append(x)
                celly_list.append(y)
                celluuid_list.append(uuid4().hex)

        cellmap_df['x_position'] = cellx_list
        cellmap_df['y_position'] = celly_list
        cellmap_df['id'] = celluuid_list

        return cellmap_df

    def initfill(self , cellmapptofill_df , seed_df, excludedcol_list = None):

        col_list = []
        if excludedcol_list :
            for col in seed_df.columns.values:
                if not col in excludedcol_list:
                    col_list.append(col)

        maplen = len(cellmapptofill_df)

        for col in col_list:
            colmean = seed_df[col].mean()
            colstd = seed_df[col].std()

            cellmapptofill_df[col] = [random.gauss(colmean, colstd) for i in range(maplen)]

        return cellmapptofill_df
