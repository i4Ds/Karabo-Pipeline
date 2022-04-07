
def setup_jupyter_env():
    """
    Needs to be run, when you want to use the pipeline inside of a jupyter notebok.
    Sets specific environment variables that the jupyter kernel is not loading by default.
    """
    from distutils.sysconfig import get_python_lib
    import os

    data_folder = f"{get_python_lib()}/../../../data"
    os.environ["RASCIL_DATA"] = data_folder
