## Operating process
### Preparing the code and data

First, you should install all the packages in the '/mp4/requirements.txt' file. 

Second, you should save the all the files that in the 'mp4' folder. You can git clone or scp all the code onto your servers.

After git clone or scp the code,you should take care of the variable 'COORDINATOR_IP' and 'STANDBY_COORDINATOR_IP' in line 47, 48 in the 'mp4_machinelearning.py' 

```
COORDINATOR_IP = "172.22.156.5"
STANDBY_COORDINATOR_IP = "172.22.94.4"
```
You can modify it into your own ip address, which is the default coordinator and stand_by coordinator ip adress.

You also cam modify the inference model's bathsize in line 45, 46 in the 'mp4_machinelearning.py' 

```
ALEXNET_BATCHSIZE = 400
RESNET_BATCHSIZE = 400
```

### Running the IDunno, a Distributed Learning Cluster System code 
Then, You can run`mp4_machinelearning.py` by typing "python3 mp4_machinelearning.py"


Once running the script, you can type different commands in the following to see the inference result and statistical data for each model(AlexNet, ResNet18).  

```
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
            "c1 see the queries rate and finished inferenced result for each model.\n"
            "c2 Show current processing time of a query in a given model (job).\n"
            "c4 see the queries result.\n"
            "cvm see on each vm what tasks are running.\n"
            "cq see how is each query distributed to (vm,start_point,end_point).\n"

```


