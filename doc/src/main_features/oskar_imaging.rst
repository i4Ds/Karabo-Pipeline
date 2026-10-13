karabo.imaging.imager_oskar
===========================

Overview
------------
This package summarizes tools and functions to be used with the imager
from the OSKAR backend. This backend does not offer functionality to
calculate a cleaned image. For cleaned images, prefer
``get_imager(ImagingBackend.SDP)`` or ``get_imager(ImagingBackend.WSCLEAN)``.
This direct dirty-imaging utility is separate from the common ``ImagingBackend``
selector. OSKAR remains a supported simulation backend; write a Measurement Set
to image its simulated visibilities with either common imaging backend.


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
