import os
import sys
import subprocess

class ZMQ_Througput(object):
    def __init__(self, params):
		self.path          = os.path.dirname(os.path.realpath(__file__))
		self.client_binary = self.path + "/" + "local_thr"
		self.server_binary = self.path + "/" + "remote_thr"
		self.run_args      = ""
		self.port          = "5555"
		self.message_size  = "1"
		self.iterations    = "1000"
		self.interface     = "eth0"

		self.__add_run_args(params)
		self.__add_hosts(params)
        self.__compile(params)

    def run_command(self, host_type):
    	command = ""
    	if host_type == "server":
    		command = self.server_binary + " tcp://" + self.interface + ":"
    	if host_type == "client":
    		command = self.client_binary + " tcp://" + self.hosts[0] + ":"

    	return command

    def execute(self):
        server_cmd  = self.run_command("server")
        client_cmd  = self.run_command("client")

        server_proc = subprocess.Popen(server_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        client_proc = subprocess.Popen(client_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = server_proc.communicate()

        if len(stderr) == 0:
            val = {} 
            val["timing"] = self.__parse_output(stdout.splitlines()[])
            val["config"] = {"server_command": server_cmd, "client_command": client_cmd}
        else:
            print("Server command: " + server_cmd)
            print("Client command: " + client_cmd)
            print("ERROR:")
            print(stderr)
            exit(1)

        return val

    def __compile(self, params=None):
        pass

    def __add_hosts(self, params):
    	# TODO: add functionality for hostfile
    	self.hosts = params["hosts"].split(",")

	def __add_run_args(self, params):
		if "port" in params:
			self.port = params["port"]
		if "message_size" in params:
		    self.message_size = params["message_size"]
		if "iterations" in params:
		    self.iterations = params["iterations"]
		if "interface" in params:
			self.interface = params["interface"]

		self.run_args = " ".join([self.port, self.message_size, self.iterations])

    def __parse_output(self, output):
    	import re

    	regex = re.compile()
        results = {}
        for line in output:
            size, t_avg, t_min, t_max, iterations = line.split()
            size = int(size)
            results[size] = {}
            results[size]["avg"] = float(t_avg)
            results[size]["min"] = float(t_min)
            results[size]["max"] = float(t_max)
            results[size]["iterations"] = int(iterations)

        return results