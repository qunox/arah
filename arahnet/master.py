from __future__ import  division
__author__ = 'qunox'

from multiprocessing.dummy import Pool
import pandas as pnd
import numpy as np
import logging
import socket
import uuid
import time
import json
import sys
import os

#=====>>CONFIG<<=======

projectpath = '/home/mmdali/Dropbox/arahnet/aranhnetpacket/master'
buffsize = 8192
endmarker = 'TAMAT'
serversproperty_dict = {


    'california':
        {'ip':'ec2-54-215-241-138.us-west-1.compute.amazonaws.com',
         'port':9995}

}

paramlist = ['leptonpT', 'leptoneta', 'leptonphi', 'missingenergymagnitude', 'missingenergyphi','jet1pt', 'jet1eta',
                 'jet1phi', 'jet1b-tag', 'jet2pt', 'jet2eta', 'jet2phi', 'jet2b-tag','jet3pt', 'jet3eta', 'jet3phi',
                 'jet3b-tag', 'jet4pt', 'jet4eta', 'jet4phi', 'jet4b-tag','m_jj','m_jjj','m_lv','m_jlv','m_bb','m_wbb',
                 'm_wwbb']
distancefunc = 'euclidean'

#==>>END OF CONFIG<<==

#=====>>FUNC<<=======
def giveid():
    return uuid.uuid4().hex


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


class multiconnect():

    def __init__(self, serverdict, log):
        self.servername = serverdict.keys()
        self.socket_list = [serverdict[name]['socketobj'] for name in serverdict.keys()]
        self.pool = Pool(processes=len(servername))
        self.log = log

    def sendjob(self, (socket,job)):
        job = job+endmarker
        socket.sendall(job)

    def recvjob(self,socket):
        return recv_end(socket)

    def multisend(self, jobdict):

        jobpacket = json.dumps(jobdict)
        self.log.info('Job Packet: %s' % jobpacket)
        if sys.getsizeof(jobpacket) > buffsize:
            raise Exception('Job packet is bigger than buff size')
        self.log.debug('Job packet: %s' % jobpacket)

        jobpacket_list = [(socket, jobpacket) for socket in self.socket_list]
        self.pool.map(self.sendjob, jobpacket_list)

    def multirecv(self):

        self.log.info('Waiting for respond')
        respond_list = self.pool.map(self.recvjob, self.socket_list)
        self.log.info('Obtain all respond')

        newrespond_list = []
        for respond in respond_list:
            dictobj = json.loads(respond)
            newrespond_list.append(dictobj)

        return newrespond_list




#=====>>END OF FUNC<<=======


print 'Bismillahhirahmanirahim'

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
filelogger = logging.FileHandler(os.path.join(projectpath, 'masterlog.txt'))
fileformat = logging.Formatter("%(asctime)s %(levelname)s: \t%(message)s")
filelogger.setFormatter(fileformat)
filelogger.setLevel(logging.DEBUG)
log.addHandler(filelogger)
log.info('Successfully create a logger')


#Creating a list of socket
log.info('Establishing connection with given address')
for servername in serversproperty_dict.keys():
    socketobj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    address = (serversproperty_dict[servername]['ip'], serversproperty_dict[servername]['port'])
    socketobj.connect_ex(address)
    serversproperty_dict[servername]['socketobj'] = socketobj
    log.debug('Connected to ip address: %s port:%s' % address)
log.info('All connection establish')
multiconnect_func = multiconnect(serversproperty_dict, log)


#loading the cellmap
log.info('Reading initial training files')
initialfolder_path = os.path.join(projectpath,'initialcondition')
cellmap_df = pnd.read_csv(os.path.join(initialfolder_path, 'initialcellmap.csv'))
trainingdict_file = open(os.path.join(initialfolder_path, 'training_dict.json'),'r')
training_dict = json.load(trainingdict_file)
trainingdict_file.close()
iterationlenght = len(training_dict['radius'])

#Initialize the client
log.info('Initializing the client')
for servername in serversproperty_dict.keys():

    jobpacket = json.dumps({'type':'initialize'})

    log.debug('Sending job to %s' % servername)
    socketobj = serversproperty_dict[servername]['socketobj']
    multiconnect_func.sendjob((socketobj, jobpacket))

    log.debug('Waiting for respond')
    respond = recv_end(socketobj)

    if respond == 'Done':
        log.debug('Successfully initialize')
    else:
        log.info("Error: Failed to initialize at %s" % servername)
        raise Exception('Server failure')
log.info('Done Initializing')


log.info('Distributing cell id')
cellid_list = cellmap_df['id'].as_matrix()
cellidgroup_list = np.array_split(cellid_list, len(serversproperty_dict.keys()))

for index in range(len(cellidgroup_list)):
    cellid_group = cellidgroup_list[index].tolist()
    servername = serversproperty_dict.keys()[index]

    log.debug('Giving id list to %s' % servername)

    jobpacket = json.dumps(
        {'type':'getidlist',
         'cellidlist': cellid_group})

    if sys.getsizeof(jobpacket) > buffsize:
        raise Exception('Job packet is bigger than buff size')

    socketobj = serversproperty_dict[servername]['socketobj']
    multiconnect_func.sendjob((socketobj, jobpacket))

    log.debug('Waiting for respond')
    respond = recv_end(socketobj)

    if respond == 'Done':
        log.debug('Successfully create worker cellmap')
    else:
        log.info("Error: Failed to initialize at %s" % servername)
        raise Exception('Server failure')

log.info('Done sending cell id')

log.info('Distributing training data')

for servername in serversproperty_dict.keys():
    log.debug('Giving data to to %s' % servername)

    jobpacket = json.dumps(
        {'type':'traindata',
         'paramlist': paramlist,
         'distancefunc': distancefunc})

    if sys.getsizeof(jobpacket) > buffsize:
        raise Exception('Job packet is bigger than buff size')

    socketobj = serversproperty_dict[servername]['socketobj']
    multiconnect_func.sendjob((socketobj, jobpacket))

    log.debug('Waiting for respond')
    respond = recv_end(socketobj)

    if respond == 'Done':
        log.debug('Successfully create worker cellmap')
    else:
        log.info("Error: Failed to initialize at %s" % servername)
        raise Exception('Server failure')

log.info('Done sending training data')

log.info('Begin Training cellmap')

jobpacket = {'type':'starttrain','id':giveid()}
log.info('Send start train command')
multiconnect_func.multisend(jobpacket)
log.info('Waiting for respond')
responddict_list = multiconnect_func.multirecv()


for irr in range(iterationlenght):

    log.info('Training iteration: %s' % irr)
    mindiff, minposition = [], []
    for respond_dict in responddict_list:
        if not respond_dict['status'] == 'Complete':
            raise Exception('Server Failure')

        mindiff.append(respond_dict['mindiff'])
        minposition.append(respond_dict['minposition'])

    minindex = mindiff.index(min(mindiff))
    jobpacket = {'type' : 'perturb',
                 'minposition' : minposition[minindex],
                 'id': giveid()}

    multiconnect_func.multisend(jobpacket)
    responddict_list = multiconnect_func.multirecv()

log.info('Done training')
log.info('Closing')

print 'Alhamdulilah'
