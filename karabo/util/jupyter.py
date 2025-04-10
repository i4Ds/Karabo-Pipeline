from IPython.core.getipython import get_ipython


def isNotebook() -> bool:
    # based on this.:
    # https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            print(
                "Detecting to be running in Jupyter Notebook"
                + "--> Settings RASCIL Environment Variable"
            )
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
