from __future__ import  division
__author__ = 'mmdali'

import pandas as pnd

class cellmaputil:

    def __init__(self): pass

    def givesourceid(self, sourcestr):

        idlist = sourcestr.strip('[').strip(']').replace("'" , "").split(',')
        idlist = ["".join(id.split()) for id in idlist]
        if idlist[0] == '': idlist = []
        return idlist

    def loadcellmap(self, path):

        try:
            cellmap_df = pnd.read_csv(path)
        except IOError:
            raise IOError('Failed to find file in project path:%s' % path)

        for col in cellmap_df.columns.values:
            if 'source_id' in col:
                newcol_list = [self.givesourceid(cellmap_df.loc[index, col])for index in cellmap_df.index.values]
                cellmap_df[col] = newcol_list

        return cellmap_df