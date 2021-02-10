from util import abstractprocess
import os
import shlex

""" This throughput test is based on these directions:
        http://zeromq.org/results:perf-howto
"""


class ZMQ_Throughput():
    def __init__(self, params):
        self.path          = shlex.quote(os.path.dirname(os.path.realpath(__file__)))
        self.client_binary = "remote_thr"
        self.server_binary = "local_thr"

    def run_command(self, host_type):
        command = []
        if host_type == "server":
            bin_path = os.path.join(self.path, self.server_binary)
            command = [bin_path] + self.server_args
        if host_type == "client":
            bin_path = os.path.join(self.path, self.client_binary)
            command = [bin_path] + self.client_args

        return command

    def execute(self, server, server_args, client, client_args):
        self.server_args   = server_args
        self.client_args   = client_args

        server_command  = self.run_command("server")
        client_command  = self.run_command("client")

        server_proc = abstractprocess.Process("ssh", host=server, command=server_command)
        client_proc = abstractprocess.Process("ssh", host=client, command=client_command)
        output = server_proc.get_output()

        if output["returncode"] != 0:
            print("Server command: " + " ".join(server_command))
            print("Client command: " + " ".join(client_command))
            print("ERROR:")
            print(output)
            exit(1)

        val = {"perf": {}} 
        val["perf"]["throughput"], val["perf"]["unit"] = self.__parse_output(output["out"].splitlines()[3])
        
        val["config"] = { "server": server, "server_command": " ".join(server_command), 
                          "client": client, "client_command": " ".join(client_command)}

        return val

    def __parse_output(self, output):
        import re

        # Format: 'mean throughput: 37.736 [Mb/s]'
        regex = re.compile("mean throughput:\s+([\d|\.]+)\s+\[(.+)\]")

        if not regex.match(output):
            print("Unexpected output:")
            print(output)
            exit(1)

        return (regex.match(output).group(1), regex.match(output).group(2))

def get_model(params):
    app = ZMQ_Throughput(params)

    return app