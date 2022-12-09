# set shared library if WSL to detect GPU drivers
import os, platform, sys

if 'WSL' in platform.release() and (os.environ.get('LD_LIBRARY_PATH') is None or 'wsl' not in os.environ['LD_LIBRARY_PATH']):
    wsl_ld_path = '/usr/lib/wsl/lib'
    if os.environ.get('LD_LIBRARY_PATH') is None:
        os.environ['LD_LIBRARY_PATH'] = wsl_ld_path
    else:
        os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']+':'+wsl_ld_path
    # Restart Python Interpreter
    # https://stackoverflow.com/questions/6543847/setting-ld-library-path-from-inside-python
    os.execv(sys.argv[0], sys.argv)
    
    
# set rascil data directory environment variable (see https://ska-telescope.gitlab.io/external/rascil/RASCIL_install.html)
from karabo.util.jupyter import set_rascil_data_directory_env
import sys

set_rascil_data_directory_env()
