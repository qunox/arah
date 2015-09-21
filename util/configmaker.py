from __future__ import division

__author__ = 'mmdali'
import json
import os
import sys

print 'Reading configuration'

config = {
            'projectpath': '/home/mmdali/Dropbox/phd/hyperparameter/control/arahnetproject',
            'winprojectpath': 'F:\\Dropbox\\arahresult\\dev',
            'sourcepath': '/home/mmdali/Dropbox/arahsample/higgs/mix/control/mini/900.csv',
            'winsourcepath': 'F:\\Dropbox\\arahsample\\higgs\\mix\\control\\mini\\900.csv',
            'secondsourcepath': '/home/mmdali/Dropbox/arahsample/higgs/mix/control/verybig/0.5.csv',
            'winsecondsourcepath': 'F:\\Dropbox\\arahsample\\higgs\\mix\\test\\mini\\900.csv',
            'wininitialtrainingpath' :  'F:\\Dropbox\\phd\\size\\1010_lowparam_logis_euclidean\\initialcondition',
            'initialtrainingpath' : '/home/mmdali/Dropbox/arahresult/phd/noise-signalmap/euclidean_3030noise_allparam_long/initialcondition',

            'loadtrainingcondition': False,
            #'training','mapping','prob_mapping'
            'process': ['training', 'mapping'],
            'usejoblib': True,

            'mapwidth': 15,
            'mapheight': 15,

            'excludedcolumn' : ['id','label'],
            'columtouse' : ['leptonpT', 'missingenergymagnitude','jet1pt',
                 'jet1b-tag', 'jet2pt','jet2b-tag','jet3pt',
                 'jet3b-tag', 'jet4pt', 'jet4b-tag','m_jj','m_jjj','m_lv','m_jlv','m_bb','m_wbb',
                 'm_wwbb'],

            #'full', 'long', 'short'
            'trainingiteration_type': 100,
            'gradient': 'auto',
            # deriviatetanh, reverselogis, logis dampedsin
            'learning_function': 'deriviatetanh',
            'learningmidpoint': 0.3,
            'radiusmidpoint': 0.4,
            #['euclidean','cityblock', 'correlation',  'chebyshev', 'cosine']
            'training_type': 'euclidean',
            'logcycle': 50,
            'plotraining': False,

            'prob_mapping_iter': 500,
            'dbs_maxdistance': 1.5,
            'dbs_minnum': 500,
            'dbs_inteactive': True
            }

if 'win32' in sys.platform:
    configfile = open('F:\\Dropbox\\arah\\config.json' , 'w')
else:
    configfile = open('/home/mmdali/Dropbox/arah/config.json' , 'w')

json.dump(config , configfile, indent=2 , sort_keys=True)
configfile.close()

print 'Done'

