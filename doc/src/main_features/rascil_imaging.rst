karabo.imaging.imager_rascil
============================

Overview
------------
This package summarizes legacy tools and functions for the RASCIL imager.
RASCIL imaging remains available during the transitional release, but it is
deprecated. New workflows should prefer the common backend interface with
``get_imager(ImagingBackend.SDP)`` or ``get_imager(ImagingBackend.WSCLEAN)``.

.. warning::

   RASCIL support is deprecated and will be removed in a future release.
   Prefer SDP for Karabo-native imaging, or WSClean when you need WSClean
   imaging through the common backend interface.


Classes
-------

.. autoclass:: karabo.imaging.imager_rascil.RascilDirtyImagerConfig
   :members:
   :special-members: __init__
   :exclude-members:


.. autoclass:: karabo.imaging.imager_rascil.RascilDirtyImager
   :members:
   :special-members: __init__
   :exclude-members: _update_header_after_resize


.. autoclass:: karabo.imaging.imager_rascil.RascilImageCleanerConfig
   :members:
   :special-members: __init__
   :exclude-members: _update_header_after_resize


.. autoclass:: karabo.imaging.imager_rascil.RascilImageCleaner
   :members:
   :special-members: __init__
   :exclude-members: _compute
