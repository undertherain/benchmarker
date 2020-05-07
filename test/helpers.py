import sys
import runpy
import os
from io import StringIO
from contextlib import redirect_stderr, redirect_stdout


def run_module(name: str, *args, run_name: str = "__main__") -> None:
    """Run a module with given parameters.

    :param name: Module name which should be run.
    :param args: List of strings, which are copied sys.argv.
    :param run_name: Entry point's name.
    """

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    sio = StringIO()
    with redirect_stdout(sio), redirect_stderr(sio):
        backup_sys_argv = sys.argv
        sys.argv = [name + ".py"] + list(args)
        runpy.run_module(name, run_name=run_name)
        sys.argv = backup_sys_argv
