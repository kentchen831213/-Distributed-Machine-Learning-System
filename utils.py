import re
import socket

INTRODUCER_HOST = "fa22-cs425-0110.cs.illinois.edu"
HOST = socket.gethostname()

class Status:
    NEW = 'RUNNING'
    RUNNING = 'RUNNING'
    LEAVE = 'LEAVE'
class Type:
    PING = "Ping"
    PONG = "Pong"
    JOIN = "Join"
    PUT = "PUT"
    GET = "GET"
    DELETE = "DELETE"
    LS = "LS"
    STORE = "STORE"
    GET_VERSIONS = "GET-VERSIONS"
    INFERENCE = "INFERENCE"
    JOB = "JOB"
    RESULT = "RESULT"


class Field:
    TYPE = "Type"
    MEMBERSHIP = "Membership"

def get_replica_neighbors(host):
    number = int(re.findall(r'01(.+).c', host)[0])
    predecessor = number - 1
    successor = number + 1
    for i in range(10):
        if successor > 10:
            yield "fa22-cs425-01%02d.cs.illinois.edu" % (successor - 10)
        else:
            yield "fa22-cs425-01%02d.cs.illinois.edu" % successor
        successor += 1

def get_neighbors(host):

    for i in range(10):
        if HOST == "fa22-cs425-01%02d.cs.illinois.edu" % (i+1):
            continue
        yield "fa22-cs425-01%02d.cs.illinois.edu" % (i+1)

def get_file_neighbors(host):

    for i in range(host,host+5):
        i = i%10
        if i == 0:
            continue
        else:
            yield "fa22-cs425-01%02d.cs.illinois.edu" % (i)

def get_all_hosts():
    l = []
    for i in range(1, 11):
        l.append("fa22-cs425-01%02d.cs.illinois.edu" % i)
    return l








ip_dict={"px6":{
"fa22-cs425-0101.cs.illinois.edu":"172.22.156.2",
"fa22-cs425-0102.cs.illinois.edu":"172.22.158.2",
"fa22-cs425-0103.cs.illinois.edu":"172.22.94.2",
"fa22-cs425-0104.cs.illinois.edu":"172.22.156.3",
"fa22-cs425-0105.cs.illinois.edu":"172.22.158.3",
"fa22-cs425-0106.cs.illinois.edu": "172.22.94.3",
"fa22-cs425-0107.cs.illinois.edu":"172.22.156.4",
"fa22-cs425-0108.cs.illinois.edu":"172.22.158.4",
"fa22-cs425-0109.cs.illinois.edu":"172.22.94.4",
"fa22-cs425-0110.cs.illinois.edu":"172.22.156.5"},
"ycc5":{
"fa22-cs425-0101.cs.illinois.edu":"172.22.156.2",
"fa22-cs425-0102.cs.illinois.edu":"172.22.158.2",
"fa22-cs425-0103.cs.illinois.edu":"172.22.94.2",
"fa22-cs425-0104.cs.illinois.edu":"172.22.156.3",
"fa22-cs425-0105.cs.illinois.edu":"172.22.158.3",
"fa22-cs425-0106.cs.illinois.edu": "172.22.94.3",
"fa22-cs425-0107.cs.illinois.edu":"172.22.156.4",
"fa22-cs425-0108.cs.illinois.edu":"172.22.158.4",
"fa22-cs425-0109.cs.illinois.edu":"172.22.94.4",
"fa22-cs425-0110.cs.illinois.edu":"172.22.156.5"}
}
