Imaging Backend Selection
=========================

Common workflow
---------------

Karabo supports SDP and WSClean imaging through ``get_imager`` and the common
``Imager`` / ``ImageSpec`` interface. Given a Karabo ``Visibility`` named ``vis``
that wraps a Measurement Set:

.. code-block:: python

   from karabo.imaging.imager_factory import ImagingBackend, get_imager
   from karabo.imaging.imager_interface import ImageSpec

   imager = get_imager(ImagingBackend.SDP)  # or ImagingBackend.WSCLEAN
   spec = ImageSpec(npix=1024, cellsize_arcsec=1.0, phase_centre_deg=(20.0, -30.0))
   dirty, psf = imager.invert(vis, spec)
   restored = imager.restore(dirty, psf)

``invert`` returns distinct dirty and PSF ``Image`` objects; ``restore`` returns
a CLEAN/restored ``Image``. Use the same imager instance for both calls. WSClean
needs the Measurement Set and image specification saved by its preceding
``invert`` call; it reruns CLEAN on those visibilities rather than deconvolving
an arbitrary dirty/PSF pair in memory.

``ImageSpec.cellsize_arcsec`` is in arcseconds per pixel. Set
``phase_centre_deg`` to the Measurement Set's phase centre: the current adapters
use the input visibility metadata and do not use this field to rephase data.
The SDP adapter currently converts to Stokes I. ``ImageSpec.polarisation`` and
``nchan`` are not general polarization/channel-selection controls in the current
adapters; check the output WCS and frequency axes.

Defaults and selection
----------------------

An explicit backend passed to ``get_imager`` takes precedence. Without one,
``parse_imaging_backend`` reads ``IMAGING_BACKEND`` and otherwise uses SDP.
The accepted imaging strings are ``"sdp"`` and ``"wsclean"`` (case-insensitive).

.. code-block:: bash

   export IMAGING_BACKEND=sdp   # or wsclean

Some scripts also expose a CLI option:

.. code-block:: bash

   python karabo/performance_test/time_karabo_reconstruction.py --imaging-backend sdp

Backend configuration
---------------------

Pass a matching configuration through the factory to control CLEAN and other
backend-specific settings. A configuration for the wrong backend raises
``TypeError``.

.. code-block:: python

   from karabo.imaging.imager_factory import SdpImagerConfig, WscleanBackendConfig

   sdp = get_imager(
       ImagingBackend.SDP,
       config=SdpImagerConfig(
           weighting="natural",
           override_cellsize=True,
           clean_niter=500,
           clean_gain=0.1,
       ),
   )
   wsclean = get_imager(
       ImagingBackend.WSCLEAN,
       config=WscleanBackendConfig(
           weighting="natural",
           clean_niter=500,
           clean_mgain=0.8,
           clean_auto_threshold=3,
       ),
   )

These are illustrative settings, not changed defaults. SDP defaults to natural
weighting and 100 Hogbom CLEAN iterations. ``override_cellsize=True`` keeps the
requested pixel scale; its default, ``False``, allows SDP to adjust it. Set
``combine_across_frequencies=False`` in ``SdpImagerConfig`` to preserve channel
images; the default combines channels by summing them.

WSClean requires its executable on ``PATH``. Its configuration defaults to
100 CLEAN iterations, a major-loop gain of 0.8, and a 3-sigma automatic
threshold. ``weighting=None`` leaves weighting to the installed WSClean's
default; select ``"natural"`` or ``"uniform"`` explicitly for reproducibility.

Comparing backends
------------------

Reuse one Measurement Set and the same image size and pixel scale. Check actual
output WCS, restoring beam, channel handling, and flux units. SDP and WSClean
have different gridders, PSF fits, CLEAN controls, and stopping criteria, so
matching iteration counts does not guarantee equivalent results. In particular,
SDP's ``clean_gain`` and WSClean's ``clean_mgain`` are different controls.

Restoring-beam ``bmaj`` and ``bmin`` and FITS ``BMAJ`` / ``BMIN`` are in degrees;
multiply by 3600 only when displaying arcseconds. ``bpa`` / ``BPA`` is in degrees.

See :doc:`base_imaging`, :doc:`sdp_imaging`, and :doc:`wsclean_imaging` for the API
reference, or :doc:`/migration_rascil` for removed entry points.
