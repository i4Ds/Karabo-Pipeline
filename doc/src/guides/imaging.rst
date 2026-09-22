Imaging guide
=============

Karabo converts a visibility data product into dirty, PSF, and restored images
through a common imaging interface. Start by selecting the backend appropriate
to the available Measurement Set and imaging requirements.

Choose a backend
----------------

The backend-selection guide explains the common ``invert`` / ``restore``
workflow, SDP and WSClean configuration, and the behavior that differs between
the adapters.

.. toctree::
   :maxdepth: 1

   /main_features/imaging_backend_selection

Next steps
----------

For class and method signatures, see the :doc:`/main_features/base_imaging`
and :doc:`/main_features/imaging` API references. The backend-specific API
pages cover SDP, WSClean, and the direct OSKAR dirty imager.
