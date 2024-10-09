"""
Copied from ska_sdp_datamodels v0.3.1 as a workaround.
We need the export_visibility_to_ms function but it isn't available until
ska_sdp_datamodels v0.2.1, while Karabo uses v0.1.3.
Upgrading to v0.2.1 will probably break RASCIL, which means we'd have to upgrade
RASCIL, which in turn means we'd have to upgrade to Python 3.10, which is out-of-scope
for now.

Additional info:
export_visibility_to_ms added to ska_sdp_datamodels in v0.2.1.
v0.2.1 and v0.2.2 would be compatible with Python 3.9.
numpy is fixed to ^1.23, <1.24. Reason stated:
"numpy version set to be compatible with RASCIL"
RASCIL and ska_sdp_datamodels seem to be quite tightly coupled, serious doubts
if RASCIL v1.0.0 would work with ska_sdp_datamodels v0.2.1 or 0.2.2.
In addition, there's a bunch of fixes to the vis_io_ms.py module that contains
export_visibility_to_ms after v0.2.2 that we probably want as well.

TODO remove karabo.util.ska_sdp_datamodels when upgrade to Python 3.10,
RASCIL > 1.0.0, ska_sdp_datamodels >= 0.2.1 or hopefully >= 0.3.1 is done.
Change imports to ska_sdp_datamodels.visibility.*
"""
import warnings
from importlib.metadata import version

from packaging.version import Version

current_version = version("ska_sdp_datamodels")
target_version = "0.2.1"
if Version(current_version) >= Version(target_version):
    warnings.warn(
        f"ska_sdp_datamodels version {current_version} is >= {target_version}. "
        "karabo.util.ska_sdp_datamodels, which was copied as a workaround, should "
        "therefore be removed and code importing it should be changed to use "
        "ska_sdp_datamodels."
    )
