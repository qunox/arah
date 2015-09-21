from __future__ import division
__author__ = 'qunox'

from scipy.spatial.distance import cdist
import pandas as pnd
import numpy as np
import logging
import socket
import time
import uuid
import json
import sys
import os

#=====>>CONFIG<<=======
print 'Loading config.json'
config_file = open('config.json', 'r')
conf_dict = json.load(config_file)
config_file.close()
projectpath = conf_dict['projectpath']
address = (conf_dict['address'][0], conf_dict['address'][1])
endmarker = 'TAMAT'
buffsize = 100024
#==>>END OF CONFIG<<==


def recv_end(the_socket):
    total_data=[];data=''
    while True:
            data=the_socket.recv(buffsize)
            if endmarker in data:
                total_data.append(data[:data.find(endmarker)])
                break
            total_data.append(data)
            if len(total_data)>1:
                #check if end_of_data was split
                last_pair=total_data[-2]+total_data[-1]
                if endmarker in last_pair:
                    total_data[-2]=last_pair[:last_pair.find(endmarker)]
                    total_data.pop()
                    break
    return ''.join(total_data)

def sendjob(socket, job):
    job = job + endmarker
    socket.sendall(job)

print 'Bismillahhirahmanirahim'

print 'Starting logger'
#creating project path
print 'Creating project path'
if not os.path.exists(projectpath):
    print 'Creating folder: ', projectpath
    os.mkdir(projectpath)
else:
    print 'Found project path at: ', projectpath

#creating logger
print 'Creating logger'
log = logging.getLogger('defaultlog')
log.setLevel(logging.DEBUG)

#console logger
consoleformat = logging.Formatter("%(asctime)s:\t%(message)s")
console = logging.StreamHandler(sys.stdout)
console.setFormatter(consoleformat)
console.setLevel(logging.INFO)
log.addHandler(console)

#file logger
filelogger = logging.FileHandler(os.path.join(projectpath, 'workerlog.txt'))
fileformat = logging.Formatter("%(asctime)s %(levelname)s: \t%(message)s")
filelogger.setFormatter(fileformat)
filelogger.setLevel(logging.DEBUG)
log.addHandler(filelogger)
log.info('Successfully create a logger')

log.info('Initializing Connection')
socketobj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socketobj.bind(address)
socketobj.listen(1)
connection, _ = socketobj.accept()
log.info("Done Initializing")

log.info("Waiting for command")
jobpacket = json.loads(recv_end(connection))

if jobpacket['type'] == 'initialize':
    initialfolder_path = os.path.join(projectpath,'initialcondition')

    log.info('Reading initial training object')
    cellmap_df = pnd.read_csv(os.path.join(initialfolder_path, 'initialcellmap.csv'))
    sourcevec_df = pnd.read_csv(os.path.join(initialfolder_path, 'samplevector.csv'))

    trainingdict_file = open(os.path.join(initialfolder_path, 'training_dict.json'),'r')
    training_dict = json.load(trainingdict_file)
    trainingdict_file.close()

    log.info('Giving done respond')
    sendjob(connection, 'Done')
else:
    raise Exception('Master-Worker Sync is lost')


log.info("Waiting for command")
jobpacket = json.loads(recv_end(connection))

if jobpacket['type'] == 'getidlist':
    log.info('Constructing workercellmap_df')

    workercellmap_df = cellmap_df.loc[cellmap_df['id'].isin(jobpacket['cellidlist'])]
    log.info('Worker cellmap length: %s' % len(workercellmap_df))
    if len(workercellmap_df) == 0:
        raise Exception('Worker cell map is zero')

    log.info('Giving done respond')
    sendjob(connection, 'Done')

else:
    raise Exception('Master-Worker Sync is lost')

log.info("Waiting for command")
jobpacket = json.loads(recv_end(connection))
log.info('Building training data')

if jobpacket['type'] == 'traindata':
    param_list = jobpacket['paramlist']
    distancefunc = jobpacket['distancefunc']
    radiusval_list = training_dict['radius']
    learningrate_list = training_dict['learningconstant']

    log.info('Giving done respond')
    sendjob(connection, 'Done')

else:
    raise Exception('Master-Worker Sync is lost')

log.info("Waiting for command")
jobpacket = json.loads(recv_end(connection))
if not jobpacket['type'] == 'starttrain':
    raise Exception('Master-Worker Sync is lost')

log.info('Start Training')
connectiontime_list = []
for irr in range(len(radiusval_list)):

    log.info('Iteration: %s Calculating minimum' % irr)
    vec = sourcevec_df.loc[irr]
    workercellmapmatrix = workercellmap_df.as_matrix(columns=param_list)
    distanceresult = cdist(workercellmapmatrix, np.array([vec]), distancefunc)
    mindiff = min(distanceresult)[0]
    minindex = np.argmin(distanceresult)
    mindfindex = workercellmap_df.index.values[minindex]
    minposition = workercellmap_df.loc[mindfindex, ['x_position', 'y_position']].as_matrix().tolist()

    workercellmap_df['distance'] = distanceresult

    log.info('Sending back result')
    respondpacket = json.dumps({
        'status':'Complete',
        'mindiff': mindiff,
        'minposition': minposition,
        'irr': irr,
        'id': uuid.uuid4().hex
    })
    sendjob(connection, respondpacket)
    starttime = time.time()

    log.info("Waiting for command")
    jobpacket = json.loads(recv_end(connection))
    connectiontime_list.append(time.time()-starttime)
    if not jobpacket['type'] == 'perturb':
        raise Exception('Master-Worker Sync is lost')

    log.info('Perturbing')
    radius = radiusval_list[irr]
    learningrate = learningrate_list[irr]
    minposition = np.array(jobpacket['minposition'])

    output_df = pnd.DataFrame(columns=param_list)
    for index in workercellmap_df.index.values:
        distance = np.linalg.norm(workercellmap_df.loc[index, ['x_position', 'y_position']].as_matrix() - minposition)
        if distance < radiusval_list[irr]:
            smoothkernel = (radiusval_list[irr]-distance)/radiusval_list[irr]
            cellvec = workercellmap_df.loc[index].as_matrix(columns=param_list)
            learnconstant = learningrate_list[irr]

            #SOM Equation here
            output = cellvec + learnconstant * smoothkernel * (vec - cellvec)
            output_df.loc[index] = output

    for index in output_df.index.values:
        workercellmap_df.loc[index, param_list] = output_df.loc[index, param_list]

    log.info('Done perturbing for irr: %s' %irr)

log.info('Done Training')

log.info('Closing connection')

connection.close()

log.info('Saving trained cellmap')
workercellmap_df.to_csv(os.path.join(projectpath,'trainedworkercellmap.csv'), index_label=False, index=False)
timefile = open(os.path.join(projectpath, 'timelist.json'), 'w')
timejson = {'timelist':connectiontime_list}
json.dump(timejson, timefile)
timefile.close()

log.info('Done')
print 'Alhamdulilah'