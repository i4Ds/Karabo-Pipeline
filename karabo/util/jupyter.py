import logging


def setup_jupyter_env():
    """
    Needs to be run, when you want to use the pipeline inside of a jupyter notebok.
    Sets specific environment variables that the jupyter kernel is not loading by default.

    This function is idempotent (running it more than once brings no side effects).

    """
    from distutils.sysconfig import get_python_lib
    import os

    data_folder = f"{get_python_lib()}/../../../data"
    os.environ["RASCIL_DATA"] = data_folder


def isNotebook():
    # based on this.:
    # https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            print("Detecting to be running in Jupyter Notebook --> Settings RASCIL Environment Variable")
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
