"""Measurement Set I/O vendored from ``ska_sdp_datamodels`` v0.3.1.

Karabo currently pins ``ska-sdp-datamodels`` v0.1.3, whose visibility package
does not yet provide the Measurement Set export API used here. Remove this
vendored package and import ``ska_sdp_datamodels.visibility`` directly once a
newer data-model release has been validated as a separate dependency upgrade.
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
