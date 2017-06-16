from mpi_base import MPI
import importlib

class OpenMPI(MPI):
    def __init__(self, flags):
        super(OpenMPI, self).__init__(flags)
        #self.default_args += " --bind-to core "

    @staticmethod
    def show_help():
        MPI.show_help()
        print("Printing OpenMPI help....\n")

    def get_mpi(self):
        command = [self.mpirun, "-np", str(self.nprocs)]
        if self.hostfile == None:
            command.extend(["-host", "localhost,localhost"])
        else:
            command.extend(["-hostfile", self.hostfile])
        
        command.extend(self.default_args)
        return command

def run(params={}):
    # TODO: move flag-parsing to the the MPI base module

    if params["misc"] == "help":
        OpenMPI.show_help()
        exit(0)

    if params["misc"] == None:
        print("Running with default MPI flags.")
        flags = {"hostfile":None, "np":2}
    else:
        flags = params["misc"].split(",")
        flags = dict(flag.split(':') for flag in flags)        

    # Setup problem
    mpi = OpenMPI(flags)

    mod       = importlib.import_module("problems." + params["problem"]+".openmpi")
    get_model = getattr(mod, 'get_model')
    app       = get_model(params)

    # setup runner: command/script, hostfile
    output = app.execute(mpi.get_mpi())

    params["time"]        = output["timing"]
    params["app_config"]  = output["config"]

    # Run ompi_info to get the version
    ompi_version = ""

    params["framework_full"] = "Open MPI" + "_" + ompi_version
    return params
