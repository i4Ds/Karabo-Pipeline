# set rascil data directory environment variable (see https://ska-telescope.gitlab.io/external/rascil/RASCIL_install.html)
from karabo.util.jupyter import set_rascil_data_directory_env
set_rascil_data_directory_env()

# set shared library if WSL to detect GPU drivers
import os, platform
if 'WSL' in platform.release():
    wsl_ld_path = ':/usr/lib/wsl/lib'
    if os.environ.get('LD_LIBRARY_PATH') is None:
        os.environ['LD_LIBRARY_PATH'] = wsl_ld_path
    else:
        os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']+wsl_ld_path
