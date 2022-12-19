import collections
import os
import socket
import time
import re
import threading
import struct
import json
import random

import utils
import logging
from logging.handlers import RotatingFileHandler
import datetime
import mp1_client
import mp1_server
import alexnet_resnet
from ftplib import FTP, _SSLSocket, error_perm
from posixpath import dirname
import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
import shutil
from statistics import mean
import linecache
import getpass

user_name = getpass.getuser()
if user_name == "ycc5":
    PORT = 8335
    SDFS_PORT = 8330
    INFERENCE_PORT = 8550
    JOB_PORT = 8660
    RESULT_PORT = 8770
else:
    PORT = 10335
    SDFS_PORT = 10330
    INFERENCE_PORT = 10550
    JOB_PORT = 10660
    RESULT_PORT = 10770
# PORT = 8335
# SDFS_PORT = 8330
# INFERENCE_PORT = 8550
# RESULT_PORT = 8660
ALEXNET_BATCHSIZE = 500
RESNET_BATCHSIZE = 400
COORDINATOR_IP = "172.22.156.5"
STANDBY_COORDINATOR_IP = "172.22.94.4"
USERNAME = "ycc5"
PASSWORD = ""
HOST = socket.gethostname()
IP = socket.gethostbyname(HOST)

SE = "<SEPARATOR>"
BUFFER_SIZE = 4096  # send 4096 bytes each time step

# should edit this for a seperate process
SEPARATE_PROCESS_HOST = "192.168.249.1"
SEPARATE_PROCESS_PORT = PORT

# define file logging info
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    filename='host.log',
                    filemode='w')
# define a handler that displays ERROR messages to the terminal
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
rotating_file_handler = RotatingFileHandler('host.log', maxBytes=102400000, backupCount=1)
logging.getLogger('').addHandler(rotating_file_handler)
recv_logger = logging.getLogger('receiver')
monitor_logger = logging.getLogger('monitor')
join_logger = logging.getLogger('join')
send_logger = logging.getLogger('send')
master_logger = logging.getLogger('master')
sdfs_logger = logging.getLogger('sdfs')


def copy_and_rename_files(fileName, newName, pathName=".", toLog=True):
    # newName = fileName + suffix
    shutil.copyfile(os.path.join(pathName, fileName), os.path.join(pathName, newName))
    if toLog:
        sdfs_logger.info(fileName + "copied as" + newName)
        print(fileName, "copied as", newName)


def sdfs_receive_file_content(s, conn_socket, filepath, sdfsfilename):
    with open(sdfsfilename, "wb") as f:
        while True:

            bytes_read = conn_socket.recv(BUFFER_SIZE)
            if not bytes_read:
                break
            f.write(bytes_read)
    print("receive file finish!")
    conn_socket.close()


def sdfs_send_file_client(filepath, sdfsfilename, s):  # s is socket
    with open(filepath, "rb") as f:
        while True:
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
                break
            s.sendall(bytes_read)
        print("file send finish!")
    s.close()


class Server:
    def __init__(self):
        timestamp = str(int(time.time()))
        # membership list, key: host, value: (timestamp, status)
        self.MembershipList = {
            HOST: (timestamp, utils.Status.NEW)}
        self.num_worker = 1
        self.alive_worker_list = []
        self.time_lock = threading.Lock()
        self.ml_lock = threading.Lock()
        self.sdfs_lock = threading.Lock()
        self.jobs_lock = threading.Lock()
        self.master = "fa22-cs425-0110.cs.illinois.edu"
        self.standbymaster = "fa22-cs425-0109.cs.illinois.edu"
        self.file_total = 0
        # record the time current process receives last ack from its neighbors
        self.last_update = {}

        self.sdfs_file_list = []  # a list to store the filename in sdfs
        self.sdfs_file_process = collections.defaultdict(list)  # key:filename, value: a list of vm this file stores in
        self.sdfs_file_version = {}  # collections.defaultdict(list)  # key:filename, value: the newest version of this file
        self.sdfs_store_dict = collections.defaultdict(list)  # key:hostname, value: a list of file this vm stores

        self.query_number = collections.defaultdict(int)  # key: alexnet, respnet  value: number of each model queries
        self.model_average_inference_time = {'alexnet': 120, 'resnet18': 120}
        self.model_inference_time_list = collections.defaultdict(list)  # {"alexnet": [], "resnet18": []}
        self.worker_set = collections.defaultdict(list)#collections.defaultdict(dict)  # key=(inference_model,query number),value=list of assigned vm
        self.inference_result_list = collections.defaultdict(list) # key:(inference_model,query_number),  value: result
        self.working_query_number_set = set()
        self.finished_query_number_set = set()
        self.working_vm_set=set()

    def join(self, toPrint=True, toRunAllTime=False, sleepTime=0.5):
        '''
        Contacts the introducer that the process will join the group and uptate its status.

        return: None
        '''
        firstTimeLoop = True
        while toRunAllTime or firstTimeLoop:
            timestamp = str(int(time.time()))
            join_logger.info("Encounter join before:")
            join_logger.info(self.MembershipList)
            self.MembershipList[HOST] = (timestamp, utils.Status.RUNNING)
            join_logger.info("Encounter after before:")
            join_logger.info(self.MembershipList)
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if toPrint:
                print("start joining")
                print(self.master)
                print(HOST)
            if HOST != self.master:
                join_msg = [utils.Type.JOIN, HOST, self.MembershipList[HOST]]
                s.sendto(json.dumps(join_msg).encode(), (self.master, PORT))
            else:
                if toPrint:
                    print("This is master host!")
            time.sleep(sleepTime)
            firstTimeLoop = False

    def send_ping(self, host):
        '''
        Send PING to current process's neighbor using UDP. If the host is leaved/failed, then do nothing.

        return: None
        '''
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while True and HOST == self.master:
            time.sleep(0.3)
            if self.MembershipList[HOST][1] == utils.Status.LEAVE or host not in self.MembershipList or \
                    self.MembershipList[host][1] == utils.Status.LEAVE:
                continue
            try:
                self.ml_lock.acquire()
                timestamp = str(int(time.time()))
                send_logger.info("Encounter send before:")
                send_logger.info(self.MembershipList)
                self.MembershipList[HOST] = (timestamp, utils.Status.RUNNING)
                send_logger.info("Encounter send after:")
                send_logger.info(self.MembershipList)

                ping_msg = [utils.Type.PING, HOST, self.MembershipList]
                s.sendto(json.dumps(ping_msg).encode(), (host, PORT))
                if host in self.MembershipList and host not in self.last_update:
                    self.time_lock.acquire()
                    self.last_update[host] = time.time()
                    self.time_lock.release()
                self.ml_lock.release()
            except Exception as e:
                print(e)

    def receiver_program(self):
        '''
        Handles receives in different situations: PING, PONG and JOIN
        When reveived PING: update membership list and send PONG back to the sender_host
        When received PONG: delete the sender_host from last_update table and update membership list
        When received JOIN: update the membershi list and notify other hosts if you are the introducer host

        return: None
        '''
        print("receiver started")
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((HOST, PORT))
        recv_logger.info('receiver program started')
        while True:
            try:
                if self.MembershipList[HOST][1] == utils.Status.LEAVE:
                    recv_logger.info("skip receiver program since " + HOST + " is leaved")
                    continue
                data, addr = s.recvfrom(4096)
                recv_logger.info("connection from: " + str(addr) + " with data: " + data.decode())
                if data:
                    request = data.decode()
                    request_list = json.loads(request)
                    sender_host = request_list[1]
                    request_type = request_list[0]

                    request_membership = request_list[2]

                    self.ml_lock.acquire()
                    if request_type == utils.Type.JOIN:
                        recv_logger.info("Encounter join before:")
                        recv_logger.info(json.dumps(self.MembershipList))

                        self.MembershipList[sender_host] = (str(int(time.time())), utils.Status.NEW)
                        recv_logger.info("Encounter join after:")
                        recv_logger.info(json.dumps(self.MembershipList))

                        if HOST == self.master:
                            recv_logger.info("introducer recv connection from new joiner: " + str(addr))

                            join_msg = [utils.Type.JOIN, sender_host, self.MembershipList[sender_host]]
                            hosts = utils.get_all_hosts()
                            ss = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                            for hostname in hosts:
                                if hostname != HOST and hostname != sender_host:
                                    ss.sendto(json.dumps(join_msg).encode(), (hostname, PORT))

                    elif request_type == utils.Type.PING:
                        recv_logger.info("Encounter PING before:")
                        recv_logger.info(json.dumps(self.MembershipList))
                        for host, value in request_membership.items():
                            timestamp, status = value[0], value[1]
                            if status == utils.Status.LEAVE:
                                self.MembershipList[host] = value

                            if host not in self.MembershipList:
                                self.MembershipList[host] = value
                                continue

                            if int(timestamp) > int(self.MembershipList[host][0]):
                                self.MembershipList[host] = (timestamp, status)
                        recv_logger.info("Encounter PING after:")
                        recv_logger.info(json.dumps(self.MembershipList))
                        pong = [utils.Type.PONG, HOST, self.MembershipList[HOST]]

                        s.sendto(json.dumps(pong).encode(), (sender_host, PORT))

                    elif request_type == utils.Type.PONG:
                        recv_logger.info("Encounter PONG before:")
                        recv_logger.info(json.dumps(self.MembershipList))
                        self.MembershipList[sender_host] = request_membership
                        if sender_host in self.last_update:
                            self.time_lock.acquire()
                            self.last_update.pop(sender_host, None)
                            self.time_lock.release()
                        recv_logger.info("Encounter PONG after:")
                        recv_logger.info(json.dumps(self.MembershipList))
                    else:
                        recv_logger.error("Unknown message type")
                    self.ml_lock.release()
            except Exception as e:
                print(e)

    def receiver_sdfs_program(self):
        '''
        Handles receives in different situations: PUT/WRITE, GET and DELETE
        When reveived PUT: update sdfs_file_list, start receive data
                           and notify other hosts to store the file if yoy are the master host
        When received GET: return the file to sender_host
        When received DELETE: delete the file from sdfs_file_list
        '''
        print("receiver file command started")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, SDFS_PORT))
        s.listen(10)
        recv_logger.info('receiver file program started')
        while True:
            try:
                if self.MembershipList[HOST][1] == utils.Status.LEAVE:
                    recv_logger.info("skip receiver program since " + HOST + " is leaved")
                    continue
                conn_socket, addr = s.accept()
                print(f"[+] {addr} is connected.")
                data = conn_socket.recv(BUFFER_SIZE)
                print(type(data))
                print(data)
                print('addr')
                # recv_logger.info("connection from: " + str(addr) + " with data: " + data)
                if data:
                    connect = data.decode()
                    request_type, sender_host, local_file, sdfs_file, version_number = connect.split(SE)

                    print("request type " + request_type)
                    print("send host " + sender_host)
                    print("local file " + local_file)
                    print("sdfs_file " + sdfs_file)

                    # self.sdfs_lock.acquire()
                    if request_type == utils.Type.PUT:
                        recv_logger.info("Encounter PUT file before:")
                        recv_logger.info(json.dumps(self.sdfs_file_list))
                        recv_logger.info("Encounter PUT file after:")
                        recv_logger.info(json.dumps(self.sdfs_file_list))

                        self.file_total += 1
                        if sdfs_file not in self.sdfs_store_dict[HOST]:
                            self.sdfs_store_dict[HOST].append(sdfs_file)
                        if (sdfs_file in self.sdfs_file_version.keys()):
                            self.sdfs_file_version[sdfs_file] += 1  # .append(1)
                        else:
                            self.sdfs_file_version[sdfs_file] = 1
                        print("receive start")
                        sdfs_receive_file_content(s, conn_socket, local_file, sdfs_file)

                        if HOST == self.master:
                            # if we are master , everytime we receive a put, we save this version with suffix .x  (x is version_number)
                            copy_and_rename_files(sdfs_file, sdfs_file + "." + str(self.sdfs_file_version[sdfs_file]))
                            print("enter host")
                            recv_logger.info("master recv connection from new file: " + str(addr))
                            self.sdfs_file_list.append(sdfs_file)
                            replica_host = abs(hash(sdfs_file)) % 10
                            start_time = time.time()
                            if self.master not in self.sdfs_file_process[sdfs_file]:
                                self.sdfs_file_process[sdfs_file].append(self.master)
                            for hostname in utils.get_file_neighbors(replica_host):

                                ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                ss.connect((hostname, SDFS_PORT))
                                ss.send(
                                    f"{request_type}{SE}{sender_host}{SE}{local_file}{SE}{sdfs_file}{SE}{version_number}".encode())
                                if hostname not in self.sdfs_file_process[sdfs_file]:
                                    self.sdfs_file_process[sdfs_file].append(hostname)
                                if sdfs_file not in self.sdfs_store_dict[hostname]:
                                    self.sdfs_store_dict[hostname].append(sdfs_file)
                                time.sleep(1)
                                sdfs_send_file_client(sdfs_file, sdfs_file, ss)
                            end_time = time.time()
                            print(end_time - start_time)

                    elif request_type == utils.Type.GET or request_type == utils.Type.GET_VERSIONS:
                        sdfs_logger.info("Receive " + connect)

                        # if it's our own get command and we are not master, we receive file from master
                        if sender_host == HOST and (not (HOST == self.master)):
                            # ss0 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            # ss0.connect(('172.22.156.5', SDFS_PORT))#receive file from master
                            # time.sleep(1)
                            ##receive whether file exists
                            '''is_file_exist_data = conn_socket.recv(BUFFER_SIZE)
                            if data:
                                is_file_exist_data_decoded = is_file_exist_data.decode()
                                request_type0, sender_host0, local_file0, sdfs_file0, version_number0 = is_file_exist_data_decoded.split(SE)'''
                            if sdfs_file == "FILE_NOT_EXIST":
                                print("File " + sdfs_file + " does not exist in the sdfs system.")
                                time.sleep(1)
                                conn_socket.close()
                                # conn_socket.close()
                            elif sdfs_file == "FILE_EXIST":
                                print("File " + sdfs_file + " exists in the sdfs system.")
                                time.sleep(1)
                                sdfs_receive_file_content(s, conn_socket, local_file,
                                                          local_file)  # save the file into local filename

                        if HOST == self.master:  # when we are master we meet get command, we send file to the sender_host
                            print("master recv:" + connect)
                            sdfs_logger.info("master recv connection from new file: " + str(addr))
                            ss1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            ss1.connect((sender_host, SDFS_PORT))
                            sdfs_file_tosend = sdfs_file

                            if (
                                    request_type == utils.Type.GET_VERSIONS):  # if get-versions we return multiple versions of merged file # TODO:why after delete, get-versions not work.
                                file_data = []
                                num_version_to_send = 0
                                if int(
                                        version_number) > 0:  # if command numversion>largest-version, we only get all versions
                                    num_version_to_send = min(int(version_number), self.sdfs_file_version[sdfs_file])
                                sdfs_file_getversions_name = sdfs_file + ".versions" + str(
                                    self.sdfs_file_version[sdfs_file]) + "to" + str(
                                    self.sdfs_file_version[
                                        sdfs_file] - num_version_to_send + 1)  # filename.versions5to2  for example
                                file_t = open(sdfs_file_getversions_name,
                                              'w').close()  # clear the content in sdfs_file_getversions_name

                                for file_ob_version in range(self.sdfs_file_version[sdfs_file],
                                                             self.sdfs_file_version[sdfs_file] - num_version_to_send,
                                                             -1):  # range(1, self.sdfs_file_version[sdfs_file] + 1):  # from latest sdfs_file_version, iterate "num_version_to_send" times
                                    # merged_fp.write("#" * 30 + "version" + str(file_ob_version) + "#" * 30 + "\n")
                                    merged_fp = open(sdfs_file_getversions_name, "ab")
                                    merged_fp.write(("#" * 30 + "version" + str(
                                        file_ob_version) + "#" * 30 + "\n").encode())  # 69Byte
                                    file_ob = sdfs_file + "." + str(file_ob_version)
                                    '''x = open(file_ob, "r")
                                    merged_fp.write(x.read())
                                    x.close()'''
                                    with open(file_ob, "rb") as f:
                                        while True:
                                            # read the bytes from the file
                                            bytes_read = f.read(BUFFER_SIZE)
                                            merged_fp.write(bytes_read)
                                            if not bytes_read:
                                                # file transmitting is done
                                                break
                                    merged_fp.close()

                                sdfs_file_tosend = sdfs_file_getversions_name

                            IS_FILE_EXIST = "FILE_EXIST"
                            if sdfs_file not in self.sdfs_file_list:
                                IS_FILE_EXIST = "FILE_NOT_EXIST"

                            ss1.send(
                                f"{request_type}{SE}{sender_host}{SE}{local_file}{SE}{IS_FILE_EXIST}{SE}{version_number}".encode())

                            time.sleep(1)
                            if sdfs_file in self.sdfs_file_list:
                                sdfs_send_file_client(sdfs_file_tosend, sdfs_file_tosend, ss1)
                            else:
                                time.sleep(1)
                                ss1.close()
                    elif request_type == utils.Type.LS:
                        if self.master == HOST:
                            ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            ss.connect((sender_host, SDFS_PORT))
                            ss.send(
                                f"{request_type}{SE}{HOST}{SE}{sdfs_file}{SE}{self.sdfs_file_process[sdfs_file]}{SE}{version_number}".encode())
                            ss.close()
                        else:
                            print(sdfs_file)
                    elif request_type == utils.Type.DELETE:
                        self.sdfs_file_list.remove(sdfs_file)
                        self.sdfs_file_process.pop(sdfs_file, None)
                        self.sdfs_file_version.pop(sdfs_file, None)
                        self.sdfs_store_dict[HOST].remove(
                            sdfs_file)  # key:hostname, value: a list of file this vm stores
                        if self.master == HOST:
                            self.sdfs_file_list.remove(sdfs_file)
                            for host in self.sdfs_file_process[sdfs_file]:
                                ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                ss.connect((host, SDFS_PORT))
                                ss.send(
                                    f"{request_type}{SE}{HOST}{SE}{sdfs_file}{SE}{sdfs_file}{SE}{version_number}".encode())
                                ss.close()

            except Exception as e:
                print(e)

    def get_num_worker(self):
        self.num_worker = 0
        for k, v in self.MembershipList.items():
            if v[1] == utils.Status.RUNNING:
                self.num_worker += 1
        return self.num_worker

    def get_alive_worker_list(self):
        self.alive_worker_list.clear()
        for k, v in self.MembershipList.items():
            if v[1] == utils.Status.RUNNING:
                self.alive_worker_list.append(k)
        return self.alive_worker_list

    def compute_average(self, inference_model):
        # TODO:compute average
        return mean(self.model_inference_time_list[inference_model])

    def assign_inference_work(self, inference_model, querynumber, start_point, end_point):
        start_point=int(start_point)
        end_point=int(end_point)
        ratio = self.model_average_inference_time["alexnet"] / self.model_average_inference_time["resnet18"]
        self.get_num_worker()
        num_worker_assigned_to_model = max(min(round(ratio / (ratio + 1) * self.num_worker), self.num_worker),
                                           0)  # 0<=num_assigned<=num_worker
        if not inference_model == "alexnet":
            num_worker_assigned_to_model = self.num_worker - num_worker_assigned_to_model
        # TODO:we know num_assigned_to_model, we want to exactly which n worker to assigns: e.g:vm1,vm3,vm5...
        num_data_to_inference = end_point - start_point + 1
        remained_num_data_to_inference = num_data_to_inference
        remained_num_worker_assigned_to_model = num_worker_assigned_to_model
        # average_data_per_worker=num_data_to_inference/num_worker_assigned_to_model
        assigned_worker_list = random.sample(self.get_alive_worker_list(),
                                             num_worker_assigned_to_model)  # 表示从[A,B]间随机生成N个数，结果以列表返回


        now_start_point = start_point
        self.worker_set[(inference_model,querynumber)].clear()
        for i, worker_id in enumerate(assigned_worker_list):
            now_average_data_per_worker = round(remained_num_data_to_inference / remained_num_worker_assigned_to_model)
            now_end_point = now_start_point + now_average_data_per_worker - 1
            self.worker_set[(inference_model,querynumber)].append((worker_id, now_start_point, now_end_point))
            remained_num_data_to_inference -= now_average_data_per_worker
            remained_num_worker_assigned_to_model -= 1
            now_start_point=now_end_point+1

        print("We assign model %s query#%s to" % (inference_model,querynumber))
        print(self.worker_set[(inference_model,querynumber)])

    def receiver_jobs_program(self):
        '''
        Handles receives "JOBS" command
        '''
        print("receiver jobs command started")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, JOB_PORT))
        s.listen(10)
        recv_logger.info('receiver inference program started')
        while True:
            try:
                conn_socket, addr = s.accept()
                print(f"[+] {addr} is connected.")
                data = conn_socket.recv(BUFFER_SIZE)
                print(type(data))
                print(data)
                print('addr')
                connect = data.decode()
                # s.send(f"{utils.Type.INFERENCE}{SE}{IP}{SE}{inferencemodel}{SE}{start_point}{SE}{end_point}{SE}{inferencemodel}{SE}{number}".encode())
                request_type, sender_host, sdfs_file, start_point, end_point, inference_model, querynumber = connect.split(
                    SE)

                print("request type: " + request_type)
                print("send host: " + sender_host)
                print("sdfs_file: " + sdfs_file)
                print("start_point: " + start_point)
                print("end_point: " + end_point)
                print("inference_model: " + inference_model)
                print("querynumber: " + querynumber)
                if self.master == HOST:#if (self.master == HOST or self.standbymaster == HOST) and to_assign_hostname==HOST:
                    self.assign_inference_work(inference_model, querynumber, start_point,
                                               end_point)  # will change self.worker_set
                    if inference_model == "alexnet":
                        batch_size = ALEXNET_BATCHSIZE
                    else:
                        batch_size = RESNET_BATCHSIZE
                    # 0-100 master assgin 5 0-20 21-40
                    # 60-160 master assign 5
                    # Because the batch size is fixed, we only need to assign different number of workers according to inference model and average time
                    for (worker_id,assigned_start_point,assigned_end_point) in self.worker_set[(inference_model,querynumber)]:
                        if worker_id==HOST:# if assigned work to master itself
                            inference_result, inference_time = alexnet_resnet.deeplearning(sdfs_file, inference_model,
                                                                                           assigned_start_point, assigned_end_point)

                            # return inference result back to master and client
                            s_return_result = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            s_return_result.connect((sender_host, RESULT_PORT))
                            s_return_result.send(
                                f"{utils.Type.RESULT}{SE}{sender_host}{SE}{sdfs_file}{SE}{assigned_start_point}{SE}{assigned_end_point}{SE}{inference_model}{SE}{querynumber}{SE}{inference_result}".encode())
                            s_return_result.close()
                            continue
                        work_start_time = time.time()
                        # self.send_inference_command(worker_id,start_point, start_point + batch_size,inference_model)
                        # I think here can directly send the inference to worker, but up to you
                        print(worker_id)
                        s_worker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s_worker.connect((utils.ip_dict[USERNAME][worker_id],
                                          JOB_PORT))  # original s.connect((worker_id, INFERENCE_PORT))
                        s_worker.send(
                            f"{utils.Type.INFERENCE}{SE}{sender_host}{SE}{sdfs_file}{SE}{assigned_start_point}{SE}{assigned_end_point}{SE}{inference_model}{SE}{querynumber}".encode())
                        s_worker.close()
                        work_end_time = time.time()
                        self.model_inference_time_list[inference_model].append(work_end_time - work_start_time)
                        # self.model_average_inference_time[inference_model] = self.compute_average(inference_model)#commented 12-3, we need this later

                else:
                    # workers get dataset
                    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    # s.connect((COORDINATOR_IP, SDFS_PORT))
                    # s.send(f"{utils.Type.GET}{SE}{HOST}{SE}{sdfs_file}{SE}{sdfs_file}{SE}{0}".encode())
                    # s.close()
                    time.sleep(3)
                    # call inference function to do inference
                    inference_result, inference_time = alexnet_resnet.deeplearning(sdfs_file, inference_model,
                                                                                   start_point, end_point)

                    # return inference result back to master and client
                    return_list = ["172.22.156.2", "172.22.158.2", "172.22.94.2", "172.22.156.3", "172.22.158.3", 
                                    "172.22.94.3", "172.22.156.4", "172.22.158.4", "172.22.94.4", "172.22.156.5"]
                    for return_host in return_list:
                        s_return_result_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s_return_result_socket.connect((return_host, RESULT_PORT))
                        s_return_result_socket.send(
                            f"{utils.Type.RESULT}{SE}{sender_host}{SE}{sdfs_file}{SE}{start_point}{SE}{end_point}{SE}{inference_model}{SE}{querynumber}{SE}{inference_result}".encode())
                        s_return_result_socket.close()
             

            except Exception as e:
                print(e)
    def receiver_inference_program(self):
        '''
        Handles receives "INFERENCE" command
        '''
        print("receiver inference command started")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, INFERENCE_PORT))
        s.listen(10)
        recv_logger.info('receiver inference program started')
        while True:
            try:
                conn_socket, addr = s.accept()
                print(f"[+] {addr} is connected.")
                data = conn_socket.recv(BUFFER_SIZE)
                print(type(data))
                print(data)
                print('addr')
                connect = data.decode()
                # f"{utils.Type.INFERENCE}{SE}{IP}{SE}{inference_model}{SE}{start_point}{SE}{end_point}".encode())
                request_type, sender_host, inference_model, start_point, end_point = connect.split(
                    SE)

                batch_size = RESNET_BATCHSIZE if inference_model == "resnet18" else ALEXNET_BATCHSIZE
                for i in range(int(start_point), int(end_point), batch_size):
                    self.send_job_command(sender_host, i, i + batch_size - 1, inferencemodel=inference_model)
                    time.sleep(20)

            except Exception as e:
                print(e)
    def receiver_inference_result_program(self):
        '''
        Handles receives "INFERENCE RESULT" command
        if receiver is cordinator: put the result tin dictionary
        if receiver is client: put the result in 
        '''
        print("receiver inference result started")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, RESULT_PORT))
        s.listen(10)
        recv_logger.info('receiver inference program started')
        #  f"{utils.Type.INFERENCE}{SE}{IP}{SE}{inference_model}{SE}{start_point}{SE}{end_point}".encode())
        while True:
            try:
                conn_socket, addr = s.accept()
                data = conn_socket.recv(BUFFER_SIZE)
                connect = data.decode()
                request_type, sender_host, sdfs_file, start_point, end_point, inference_model, querynumber, inference_result= connect.split(SE)
                self.inference_result_list[str(inference_model)+" " +str(querynumber)].append(inference_result)
                

                
            except Exception as e:
                pass

    def monitor_program(self):
        '''
        Monitor daemon that checks if any neighbor process has timeout

        If the timeout process it the master, than choose another master and send it to other processes

        return: None
        '''
        print("monitor started")
        while True and HOST == self.master:
            try:
                self.time_lock.acquire()

                keys = list(self.last_update.keys())
                for hostname in keys:
                    if time.time() - self.last_update[hostname] > 2:
                        value = self.MembershipList.get(hostname, "*")
                        if value != "*" and value[1] != utils.Status.LEAVE:
                            self.MembershipList[hostname] = (value[0], utils.Status.LEAVE)#now we detect a machine fail."hostname" fails
                            for file in self.sdfs_store_dict[hostname]:
                                self.sdfs_file_process[file].remove(hostname)

                                for replica_host in utils.get_replica_neighbors(hostname):
                                    if replica_host in self.sdfs_file_process[file] \
                                            or self.MembershipList[replica_host][1] == utils.Status.LEAVE:
                                        continue

                                    ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                    print(replica_host)
                                    try:
                                        ss.connect((replica_host, SDFS_PORT))
                                        ss.send(f"{utils.Type.PUT}{SE}{HOST}{SE}{file}{SE}{file}{SE}{1}".encode())
                                        self.sdfs_file_process[file].append(replica_host)
                                        self.sdfs_store_dict[replica_host].append(file)
                                    except socket.error as exc:
                                        continue

                                    time.sleep(1)
                                    sdfs_send_file_client(file, file, ss)
                                    break

                            self.sdfs_store_dict.pop(hostname, None)

                            monitor_logger.info("Encounter timeout after:")
                            monitor_logger.info(json.dumps(self.MembershipList))

                        self.last_update.pop(hostname, None)

                self.time_lock.release()
            except Exception as e:
                print(e)

    def send_sdfs_command(self, host, port, command, localfile="no", sdfsfile="no", version_number=-1):
        '''
        Send sdfs command to current process's neighbor using TCP. If the host is leaved/failed, then do nothing.
        return: None
        '''
        print("sdfs sender started")

        if self.MembershipList[HOST][1] == utils.Status.LEAVE or host not in self.MembershipList or \
                self.MembershipList[host][1] == utils.Status.LEAVE:
            pass
        else:
            try:
                timestamp = str(int(time.time()))
                sdfs_logger.info("sdfs Encounter send before:")
                sdfs_logger.info(self.sdfs_file_list)

                if command == utils.Type.PUT:
                    if self.master == HOST:
                        self.file_total += 1
                        self.sdfs_store_dict[HOST].append(sdfsfile)
                        if (sdfsfile in self.sdfs_file_version.keys()):
                            self.sdfs_file_version[sdfsfile] += 1  # .append(1)
                        else:
                            self.sdfs_file_version[sdfsfile] = 1
                        ##
                        self.sdfs_file_list.append(sdfsfile)
                        replica_host = abs(hash(sdfsfile)) % 10
                        for hostname in utils.get_file_neighbors(replica_host):
                            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            s.connect((hostname, port))
                            s.send(f"{command}{SE}{HOST}{SE}{localfile}{SE}{sdfsfile}{SE}{version_number}".encode())
                            self.sdfs_file_process[sdfsfile].append(hostname)
                            time.sleep(1)
                            sdfs_send_file_client(localfile, sdfsfile, s)
                    else:
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s.connect(('172.22.156.5', port))
                        s.send(f"{command}{SE}{HOST}{SE}{localfile}{SE}{sdfsfile}{SE}{version_number}".encode())
                        time.sleep(1)
                        sdfs_send_file_client(localfile, sdfsfile, s)
                elif command == utils.Type.GET or command == utils.Type.GET_VERSIONS:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect(('172.22.156.5', port))
                    s.send(f"{command}{SE}{HOST}{SE}{localfile}{SE}{sdfsfile}{SE}{version_number}".encode())
                    s.close()
                elif command == utils.Type.LS:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect(('172.22.156.5', port))
                    s.send(f"{command}{SE}{HOST}{SE}{sdfsfile}{SE}{sdfsfile}{SE}{version_number}".encode())
                    s.close()
                elif command == utils.Type.DELETE:

                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect(('172.22.156.5', port))
                    s.send(f"{command}{SE}{HOST}{SE}{sdfsfile}{SE}{sdfsfile}{SE}{version_number}".encode())
                    s.close()
                sdfs_logger.info("sdfs Encounter send after:")
                sdfs_logger.info(self.sdfs_file_list)
            except Exception as e:
                print(e)

    def send_job_command(self, sender_host, start_point, end_point, inferencemodel="no"):
        '''
        Send inference command to current coordinator.
        return: None
        '''

        if self.MembershipList[HOST][1] == utils.Status.LEAVE or HOST not in self.MembershipList:
            pass
        else:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.connect((COORDINATOR_IP, JOB_PORT))
            except:
                s.connect((STANDBY_COORDINATOR_IP, JOB_PORT))
            time.sleep(1)
            self.query_number[inferencemodel] += 1
            number = self.query_number[inferencemodel]
            s.send(
                f"{utils.Type.JOB}{SE}{sender_host}{SE}{inferencemodel}{SE}{start_point}{SE}{end_point}{SE}{inferencemodel}{SE}{number}".encode())
        s.close()

    def monitor_master(self):
        print("monitor master started")

    def leave(self):
        '''
        Mark current process as LEAVE status

        return: None
        '''
        self.time_lock.acquire()
        prev_timestamp = self.MembershipList[HOST][0]
        monitor_logger.info("Encounter leave before:")
        monitor_logger.info(json.dumps(self.MembershipList))
        self.MembershipList[HOST] = (prev_timestamp, utils.Status.LEAVE)
        monitor_logger.info("Encounter leave after:")
        monitor_logger.info(json.dumps(self.MembershipList))
        print(self.MembershipList)
        self.time_lock.release()

    def print_membership_list(self):
        '''
        Print current membership list

        return: None
        '''
        print(self.MembershipList)

    def print_self_id(self):
        '''
        Print self's id

        return: None
        '''
        print(IP + "#" + self.MembershipList[HOST][0])

    def put(self, localfilename, sdfsfilename):

        self.send_sdfs_command(self.master, SDFS_PORT, utils.Type.PUT, localfilename, sdfsfilename)

    def get(self, sdfsfilename, localfilename):
        if HOST == self.master:
            copy_and_rename_files(sdfsfilename, localfilename)  # , pathName=".", toLog=True)
        else:
            self.send_sdfs_command(self.master, SDFS_PORT, command=utils.Type.GET, sdfsfile=sdfsfilename,
                                   localfile=localfilename)

    def delete(self, sdfsfilename):
        if HOST == self.master:
            self.sdfs_file_list.remove(sdfsfilename)
            self.sdfs_file_process[sdfsfilename] = []
            self.sdfs_file_version.pop(sdfsfilename, "This file has been deleted")
        else:
            self.send_sdfs_command(self.master, SDFS_PORT, command="DELETE", sdfsfile=sdfsfilename)

    def ls(self, sdfsfilename):
        if HOST == self.master:
            print(sdfsfilename + " is in the following VMs:")
            print(self.sdfs_file_process[sdfsfilename])
        else:
            self.send_sdfs_command(self.master, SDFS_PORT, command="LS", sdfsfile=sdfsfilename)

    def store(self):
        print("sdfs_store_dict" + HOST + "=")
        print(self.sdfs_store_dict[HOST])

    def get_versions(self, sdfsfilename, num_versions, localfilename):
        self.send_sdfs_command(self.master, SDFS_PORT, command=utils.Type.GET_VERSIONS, sdfsfile=sdfsfilename,
                               version_number=num_versions, localfile=localfilename)

    def inference(self, start_point, end_point, inference_model):

        '''
        Send inference command to current coordinator.
        return: None
        '''
        if self.MembershipList[HOST][1] == utils.Status.LEAVE or HOST not in self.MembershipList:
            pass
        else:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.connect((COORDINATOR_IP, INFERENCE_PORT))
            except:
                s.connect((STANDBY_COORDINATOR_IP, INFERENCE_PORT))
            time.sleep(1)
            s.send(
                f"{utils.Type.INFERENCE}{SE}{IP}{SE}{inference_model}{SE}{start_point}{SE}{end_point}".encode())
        s.close()


    def shell(self):
        print(
            "Welcome to the interactive shell for CS425 MP2. You may press 1/2/3/4 or input command for below functionalities.\n"
            "1. list_mem: list the membership list\n"
            "2. list_self: list self's id\n"
            "3. join: command to join the group\n"
            "4. leave: command to voluntarily leave the group (different from a failure, which will be Ctrl-C or kill)\n"
            "5. list_master: list master\n"
            "6. grep: get into mp1 grep program\n"
            "7. put localfilename sdfsfilename: upload localfilename from local dir to sdfs\n"
            "8. get sdfsfilename localfilename: get file from sdfs to local\n"
            "9. delete sdfsfilename: delete file from sdfs\n"
            "10.ls sdfsfilename: list all machine (VM) addresses where this file is currently being stored\n"
            "11.store: At any machine, list all files currently being stored at this machine.\n"
            "12.get-versions sdfsfilename num-versions localfilename:gets all the last num-versions versions of the file into the localfilename (use delimiters to mark out versions).\n"
            "13 inference, start point ,end point, inferen_model name(alexnet or resnet).\n"
            "c4 see the queries result.\n"
            )

        time.sleep(1)
        # interactive shell
        while True:
            input_str = input("Please enter input: ")
            splited_input_str = input_str.split()
            if input_str == 'exit':
                break
            if input_str == "1":
                print("Selected list_mem")
                self.print_membership_list()
            elif input_str == "2":
                print("Selected list_self")
                self.print_self_id()
            elif input_str == "3":
                print("Selected join the group")
                self.join()
            elif input_str == "4":
                print("Selected voluntarily leave")
                self.leave()
            elif input_str == "5":
                print("Selected list master")
                print(self.master)
            elif input_str == "6":
                input_command = input("Please enter grep command: ")
                c = mp1_client.Client(input_command)
                t = threading.Thread(target=c.query)
                t.start()
                t.join()
            elif input_str.startswith("put "):
                if len(splited_input_str) == 3:
                    self.put(splited_input_str[1], splited_input_str[2])
                else:
                    print("Error: missing or too many " + splited_input_str[0] + " parameter .")
            elif input_str.startswith("get "):
                if len(splited_input_str) == 3:
                    self.get(splited_input_str[1], splited_input_str[2])
                else:
                    print("Error: missing or too many " + splited_input_str[0] + " parameter .")
            elif input_str.startswith("delete "):
                if len(splited_input_str) == 2:
                    self.delete(splited_input_str[1])
                else:
                    print("Error: missing or too many " + splited_input_str[0] + " parameter .")
            elif input_str.startswith("ls "):
                if len(splited_input_str) == 2:
                    self.ls(splited_input_str[1])
                else:
                    print("Error: missing or too many " + splited_input_str[0] + " parameter .")
            elif input_str.startswith("store"):
                if len(splited_input_str) == 1:
                    self.store()
                else:
                    print("Error: missing or too many " + splited_input_str[0] + " parameter .")
            elif input_str.startswith("get-versions "):
                if len(splited_input_str) == 4 and int(splited_input_str[2]) > 0:
                    self.get_versions(splited_input_str[1], splited_input_str[2], splited_input_str[3])
                elif len(splited_input_str) == 4 and int(splited_input_str[2]) <= 0:
                    print("Error: num-versions should greater than 0.")
                else:
                    print("Error: missing or too many " + splited_input_str[0] + " parameter .")
            elif input_str.startswith("inference"):
                if len(splited_input_str) == 4:
                    self.inference(splited_input_str[1], splited_input_str[2], splited_input_str[3])
                else:
                    print("Error: missing or too many " + splited_input_str[0] + " parameter .")
            elif input_str.startswith("c4"):
                with open('result.txt', 'w') as convert_file:
                    convert_file.write(json.dumps(self.inference_result_list))
                print(self.inference_result_list)
                
            else:
                print("Invalid input. Please try again")

    def run(self):
        '''
        run function starts the server

        return: None
        '''
        logging.info('Enter run() function.')
        t_monitor = threading.Thread(target=self.monitor_program)
        t_receiver = threading.Thread(target=self.receiver_program)
        t_file_receiver = threading.Thread(target=self.receiver_sdfs_program)
        t_inference_receiver = threading.Thread(target=self.receiver_inference_program)
        t_jobs_receiver = threading.Thread(target=self.receiver_jobs_program)
        t_inference_result_receiver = threading.Thread(target=self.receiver_inference_result_program)
        t_shell = threading.Thread(target=self.shell)
        t_server_mp1 = threading.Thread(target=mp1_server.server_program)
        threads = []
        i = 0
        for host in utils.get_neighbors(HOST):
            t_send = threading.Thread(target=self.send_ping, args=(host,))
            threads.append(t_send)
            i += 1
        t_monitor.start()
        t_receiver.start()
        t_file_receiver.start()
        t_inference_receiver.start()
        t_jobs_receiver.start()
        t_inference_result_receiver.start()
        t_shell.start()
        t_server_mp1.start()
        for t in threads:
            t.start()
        t_monitor.join()
        t_receiver.join()
        t_file_receiver.join()
        t_inference_receiver.join()
        t_jobs_receiver.join()
        t_inference_result_receiver.join()
        t_shell.join()
        t_server_mp1.join()
        for t in threads:
            t.join()


if __name__ == '__main__':
    s = Server()
    s.run()
