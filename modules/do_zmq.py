import os
import sys
import importlib

class ZMQ(object):
    def __init__(self, flags):
        self.run_args      = ""
        self.message_size  = "1"
        self.iterations    = "1000"
        self.interface     = "ib0"
        self.server_port   = "5555"

        self.__add_run_args(flags)

    def __add_run_args(self, flags):
        if "svr_ip" in flags:
            self.server = flags["svr_ip"]
        if "clt_ip" in flags:
            self.client = flags["clt_ip"]

        if "port" in flags:
            self.server_port = flags["port"]
        if "message_size" in flags:
            self.message_size = flags["message_size"]
        if "iterations" in flags:
            self.iterations = flags["iterations"]
        if "interface" in flags:
            self.interface = flags["interface"]

        self.run_args = [self.message_size, self.iterations]

    def get_server_args(self):
        binding = "tcp://" + self.interface + ":" + self.server_port
        command = [binding]
        command.extend(self.run_args)

        return command

    def get_client_args(self):
        binding = "tcp://" + self.server + ":" + self.server_port
        command = [binding]
        command.extend(self.run_args)

        return command

    @staticmethod
    def show_help():
        print("Printing ZMQ help....\n")
        print("Example:")
        print("     # python3.6 ./benchmarker.py --framework=zmq --problem=bandwidth --misc=svr_ip:10.1.28.83,clt_ip:10.1.28.82,interface:ib0")

def run(params):
    params["problem"] = "bandwidth"

    # Prepare paramters
    if params["misc"] == "help":
        ZMQ.show_help()
        exit(0)

    if params["misc"] == None:
        print("There are no ZMQ flags provided.\nPlease use --misc=<flags>.")
        ZMQ.show_help()
        exit(1)

    flags = params["misc"].split(",")
    flags = dict(flag.split(':') for flag in flags)        

    # Setup benchmark
    zmq = ZMQ(flags)

    mod       = importlib.import_module("problems." + params["problem"]+".zmq")
    get_model = getattr(mod, 'get_model')
    app       = get_model(params)

    # run actual benchmark and measure performance
    output = app.execute(zmq.server, zmq.get_server_args(), zmq.client, zmq.get_client_args())
    params["perf"]        = output["perf"]
    params["app_config"]  = output["config"]

    # Find the version of ZMQ
    zmq_version = ""

    params["framework_full"] = "ZMQ" + "_" + zmq_version
    return params