Common Imaging API
==================

Use the factory and shared interface for backend-switched imaging. See
:doc:`imaging_backend_selection` for examples and current adapter limitations.

Interface and factory
---------------------

.. autoclass:: karabo.imaging.imager_interface.ImageSpec
   :members:

.. autoclass:: karabo.imaging.imager_interface.Imager
   :members:

.. autoclass:: karabo.imaging.imager_factory.ImagingBackend
   :members:
   :undoc-members:

.. autofunction:: karabo.imaging.imager_factory.parse_imaging_backend

.. autofunction:: karabo.imaging.imager_factory.get_imager

Direct-imager base classes
-----------------------------------

These lower-level classes support the direct OSKAR dirty imager and the direct
WSClean implementation. They are not the backend-neutral ``Imager`` interface.
Their ``imaging_cellsize`` parameter uses radians; ``ImageSpec`` uses arcseconds.

.. autoclass:: karabo.imaging.imager_base.DirtyImagerConfig
   :members:
   :special-members: __init__

.. autoclass:: karabo.imaging.imager_base.DirtyImager
   :members:
   :special-members: __init__

.. autoclass:: karabo.imaging.imager_base.ImageCleanerConfig
   :members:
   :special-members: __init__

.. autoclass:: karabo.imaging.imager_base.ImageCleaner
   :members:
   :special-members: __init__
