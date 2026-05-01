karabo.imaging.imager_wsclean
=============================

Overview
------------
This package summarizes tools and functions to be used with the imager
based on the WSClean algorithm.

For normal imaging workflows, prefer the common backend interface:
``get_imager(ImagingBackend.WSCLEAN)``. The direct classes documented here remain
available for compatibility and custom WSClean commands, but emit a warning.


Classes
-------

.. autoclass:: karabo.imaging.imager_wsclean.WscleanDirtyImager
   :members:
   :special-members: __init__
   :exclude-members: 


.. autoclass:: karabo.imaging.imager_wsclean.WscleanImageCleanerConfig
   :members:
   :special-members: __init__
   :exclude-members:


.. autoclass:: karabo.imaging.imager_wsclean.WscleanImageCleaner
   :members:
   :special-members: __init__
   :exclude-members:


Functions
---------

.. autofunction:: karabo.imaging.imager_wsclean.create_image_custom_command
