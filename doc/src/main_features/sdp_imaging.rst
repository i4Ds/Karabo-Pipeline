SDP Imaging
===========

Select SDP with ``get_imager(ImagingBackend.SDP, config=SdpImagerConfig(...))``.
Import the configuration from ``karabo.imaging.imager_factory``; users do not
need to instantiate the implementation class below.

The adapter reads a Measurement Set, converts its visibilities to Stokes I,
flags autocorrelations, and performs separate dirty and PSF inversions.
Restoration uses Hogbom CLEAN followed by convolution with the fitted clean
beam and addition of the residual. Other CLEAN algorithms currently raise
``NotImplementedError``. After restoration, ``last_model_image`` and
``last_residual_image`` expose the exported model and residual on the SDP imager.

See :doc:`imaging_backend_selection` for pixel-scale and channel-combination
behavior, and :doc:`/migration_rascil` for upgrading older imaging scripts.

Configuration
-------------

.. autoclass:: karabo.imaging.backends.sdp_backend.SdpImagerConfig
   :members:
   :undoc-members:

Implementation reference
------------------------

.. autoclass:: karabo.imaging.backends.sdp_backend.SdpImager
   :members: invert, restore
