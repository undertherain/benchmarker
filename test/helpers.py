import sys
import runpy


def run_module(name: str, *args, run_name: str = "__main__") -> None:
    """Run a module with given parameters.

    :param name: Module name which should be run.
    :param args: List of strings, which are copied sys.argv.
    :param run_name: Entry point's name.
    """
    backup_sys_argv = sys.argv
    sys.argv = [name + ".py"] + list(args)
    runpy.run_module(name, run_name=run_name)
    sys.argv = backup_sys_argv
