karabo.imaging.imager_oskar
===========================

Overview
------------
This package summarizes tools and functions to be used with the imager
from the OSKAR backend. This backend does not offer functionality to
calculate a cleaned image. For cleaned images, prefer
``get_imager(ImagingBackend.SDP)`` or ``get_imager(ImagingBackend.WSCLEAN)``.
RASCIL remains available only as a deprecated legacy option.


Classes
-------

.. autoclass:: karabo.imaging.imager_oskar.OskarDirtyImagerConfig
   :members:
   :special-members: __init__
   :exclude-members: 


.. autoclass:: karabo.imaging.imager_oskar.OskarDirtyImager
   :members:
   :special-members: __init__
   :exclude-members:
