## What this script is for
The dishes of the ALMA telescope (Atacama Large Microwave Array) can operate in different configurations. These 'cycles' are set up at different points in time (see link [1]). For the next year (2025) three new cycles are added to the configuration: cycle 9, 10 and 11. This script can be used to download the latest or update the current configuration files.

The files are fetched directly from the ALMA server. The url is not under our control. Therefore, the url may change which breaks the code. In this case have a look at link [2] and update the variable `CONFIG_FILE_URL` in the code.

##  Setup
1. Create and activate a Python virtual env / conda env / similar
2. `pip install requests` should be enough

## Important links:
1. The ALMA configuration schedule: https://almascience.eso.org/news/the-alma-configuration-schedule-for-cycles-9-10-and-11
2. The configuration files: https://almascience.eso.org/tools/casa-simulator
