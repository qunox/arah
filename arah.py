from __future__ import division

__author__ = 'mmdali'

import os
import json
import pandas as pnd

from sys import argv

from util import configreader
from util import logger, cellmaputil
from src import cellmap, dfutil, pretraining, mapping2, mapanalysis, training3

# The main

if __name__ == '__main__':

    print '=' * 80
    print 'Bismillahhirahmanirahim'.center(80)
    print 'Arah version : 0.1'.center(80)
    print ' <<---------------'.center(80)
    print '=' * 80

    if len(argv) == 2:
        configfilepath = argv[1]
    elif len(argv) == 1:
        '-->Default configuration file will be use'
        configfilepath = os.path.join(os.getcwd(), 'config.json')
    else:
        raise Exception('ERROR: Unexpected number or argument passed')

    config_func = configreader.config()
    config = config_func.giveconfig(configfilepath)

    if not os.path.exists(config.projectpath):
        os.mkdir(config.projectpath)

    if 'dev' not in config.processlist:
        config_func.saveconfig()

    print '-->Creating logging file'
    log_func = logger.logger()
    log = log_func.givelogger(config.logfile)
    log.info('\n')
    log.info(' START '.center(48,'='))
    log.info('Arah logger has been successfully  started')

    log.info('Reading source file and creating a data frame')
    log.debug('Source path: %s' % config.sourecepath)
    primerawsource_df = pnd.read_csv(config.sourecepath)
    log.info('Raw source data frame created')


    if config.secondsource:
        log.info('Reading second source file and creating its data frame')
        log.debug('Secondary source path: %s' % config.secondsourcepath)
        secondrawsource_df = pnd.read_csv(config.secondsourcepath)
        log.info('Raw second source data frame created')

    if 'training' in config.processlist:

        if config.loadtrainingcondition is True:
            log.info('Loading initial training condition')
            log.warning('WARNING: Training configuration specify in config file will not be use')
            cellmap_df = pnd.read_csv(os.path.join(config.initialtrainingconditionpath, 'initialcellmap.csv'))
            samplevec_df = pnd.read_csv(os.path.join(config.initialtrainingconditionpath, 'samplevector.csv'))
            trainingdict_file = open(os.path.join(config.initialtrainingconditionpath, 'training_dict.json'), 'r')
            training_dict = json.load(trainingdict_file)
            trainingdict_file.close()
            learningconst_list = training_dict['learningconstant']
            radius_list = training_dict['radius']
            log.info('Finish loading intial condition')

        else:
            log.info('Entering training process')
            log.info('Creating cell-map data frame')
            cellmap_func = cellmap.cellmap()
            cellmap_df = cellmap_func.givecellmap(config.mapwidth, config.mapheight)
            cellmap_df = cellmap_func.initfill(cellmap_df, primerawsource_df, excludedcol_list=config.excludedcol)

            log.info('Pre-mapping preparation')
            log.debug('Preparing sample vector')
            pretraining_func = pretraining.pretraining()
            config.trainingiteration = pretraining_func.givetraininglength(config.trainingiteration_type,
                                                                           len(primerawsource_df))
            samplevec_df = pretraining_func.givesamplevec(primerawsource_df, config.trainingiteration,
                                                          config.trainingiteration_type, excludedcolumn=config.excludedcol)

            log.debug('Done preparing sample vector')
            log.debug('Preparing learning constant')
            learningconst_list = pretraining_func.givelearningconstant(config.trainingiteration, x0=config.learnmidpoint,
                                                                       plot=True, projectpath=config.projectpath,
                                                                       k=config.gradient, type=config.learningfunc)

            log.debug('Done preparing learning constant')
            log.info('Preparing influence radius')
            radius_list = pretraining_func.giveradius(config.trainingiteration, 0.8, config.mapwidth, config.mapheight,
                                                      x0=config.radiusmidpoint, plot=True, projectpath=config.projectpath,
                                                      k=config.gradient)

            log.debug('Done preparing influence radius')
            log.info('Done pre-training preparations')

            log.info('Saving initial conditions')
            initialdump_path = os.path.join(config.projectpath, 'initialcondition')
            if not os.path.exists(initialdump_path): os.mkdir(initialdump_path)
            log.debug('Saving cellmap_df')
            cellmap_df.to_csv(os.path.join(initialdump_path, 'initialcellmap.csv'), index=False, index_label=False)
            log.debug('Saving sample vector df')
            samplevec_df.to_csv(os.path.join(initialdump_path, 'samplevector.csv'), index=False, index_label=False)
            log.debug('Saving learning constant and radius list')
            training_dict = {'learningconstant' : learningconst_list, 'radius' : radius_list}
            trainingdict_file = open(os.path.join(initialdump_path, 'training_dict.json'), 'w')
            json.dump(training_dict, trainingdict_file, indent=2 , sort_keys=True)
            trainingdict_file.close()
            log.debug('Finish saving intial training condition')

        log.info('Beging transforming cell map')
        training_func = training3.training(samplevec_df, cellmap_df, radius_list, learningconst_list, config.paramlist,
                                           log=log, plot=config.plottraining,project_dir=config.projectpath, cpucount=-1,
                                           type=config.trainingtype, logcycle=config.logcycle)
        training_func.starttraining()
        log.info('Getting trained cell map')
        traicellmap = training_func.givecellmap()

        log.info('Saving cell map in csv form')
        cellmapdumpfile = os.path.join(config.projectpath, 'trainedcellmap.csv')
        log.debug('Saving to: %s' % cellmapdumpfile)
        traicellmap.to_csv(cellmapdumpfile, index_label=False, index=False)
        log.info('Finish Saving cell map in csv form')

    if 'mapping' in config.processlist:

        if not 'cellmap_df' in locals():
            log.info('Loading cellmap data frame from project file: %s' % config.projectpath)
            cellmaputil_func = cellmaputil.cellmaputil()
            cellmap_df = cellmaputil_func.loadcellmap(os.path.join(config.projectpath, 'trainedcellmap.csv'))

        log.info('Begin mapping source to cell map')
        #primary
        if 'dev' in config.processlist:
            import random
            _srtscalledsource_df = primerawsource_df.ix[ random.sample(primerawsource_df.index , 1000)]
            mapping_func = mapping2.mapping(cellmap_df, _srtscalledsource_df, config.projectpath, config.paramlist ,
                                           plot=True, log=log, type=config.trainingtype)
        else:
            mapping_func = mapping2.mapping(cellmap_df, primerawsource_df, config.projectpath, config.paramlist,
                                           plot=True, log=log, type=config.trainingtype)

        mappedcellmap_df = mapping_func.mapsource()

        #secondary
        if config.secondsource:
            if 'dev' in config.processlist:
                import random
                _srtscalledsource_df = secondrawsource_df.ix[random.sample(secondrawsource_df.index , 1000)]
                mapping_func = mapping2.mapping(cellmap_df, _srtscalledsource_df, config.projectpath,
                                               config.paramlist, plot=True, log=log, secondary=True, type=config.trainingtype)
            else:
                mapping_func = mapping2.mapping(cellmap_df, secondrawsource_df, config.projectpath, config.paramlist,
                                               plot=True, log=log, secondary=True, type=config.trainingtype)

        mappedcellmap_df = mapping_func.mapsource()
        log.info('Saving mapped cell map')
        mappedcellmap_filepath = os.path.join(config.projectpath, 'mappedcell.csv')
        mappedcellmap_df.to_csv(mappedcellmap_filepath, index_label=False, index=False)


    if 'prob_mapping' in config.processlist:

        if not 'cellmap_df' in locals():
            log.info('Loading cellmap data frame from project file: %s' % config.projectpath)
            cellmaputil_func = cellmaputil.cellmaputil()
            cellmap_df = cellmaputil_func.loadcellmap(os.path.join(config.projectpath, 'mappedcell.csv'))

        log.info('Begin map analysis, please wait...')
        mappedcellmap_filepath = os.path.join(config.projectpath, 'mappedcell.csv')
        mapanalysis_func = mapanalysis.mapanalysis(cellmap_df, config.paramlist, log=log,)
        cellmap_df = mapanalysis_func.probmap(iteration=config.probmapiter)
        cellmap_df.to_csv(mappedcellmap_filepath, index_label=False, index=False)
        mapanalysis_func.densmap2(config.projectpath, coltouse='secondary', minstd=3, plot=True)
        log.info('Saving cell map')
        cellmap_df.to_csv(mappedcellmap_filepath, index_label=False, index=False)

    log.info('END\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>END<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')

