Common Imaging API
==================

Use the factory and shared interface for backend-switched imaging. See
:doc:`imaging_backend_selection` for examples and current adapter limitations.

Interface and factory
---------------------

.. autoclass:: karabo.imaging.imager_interface.ImageSpec
   :no-index:
   :members:

.. autoclass:: karabo.imaging.imager_interface.Imager
   :no-index:
   :members:

.. autoclass:: karabo.imaging.imager_factory.ImagingBackend
   :no-index:
   :members:
   :undoc-members:

.. autofunction:: karabo.imaging.imager_factory.parse_imaging_backend
   :no-index:

.. autofunction:: karabo.imaging.imager_factory.get_imager
   :no-index:

Direct-imager base classes
-----------------------------------

These lower-level classes support the direct OSKAR dirty imager and the direct
WSClean implementation. They are not the backend-neutral ``Imager`` interface.
Their ``imaging_cellsize`` parameter uses radians; ``ImageSpec`` uses arcseconds.

.. autoclass:: karabo.imaging.imager_base.DirtyImagerConfig
   :no-index:
   :members:
   :special-members: __init__

.. autoclass:: karabo.imaging.imager_base.DirtyImager
   :no-index:
   :members:
   :special-members: __init__

.. autoclass:: karabo.imaging.imager_base.ImageCleanerConfig
   :no-index:
   :members:
   :special-members: __init__

.. autoclass:: karabo.imaging.imager_base.ImageCleaner
   :no-index:
   :members:
   :special-members: __init__
