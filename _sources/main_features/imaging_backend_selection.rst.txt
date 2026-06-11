Imaging Backend Selection
=========================

Overview
--------
Karabo supports multiple imaging backends while migrating from RASCIL-based
imaging to SKA-SDP imaging. For reproducible workflows, pass the backend
explicitly in code rather than relying on environment defaults.

For backend-switched imaging, use:

- ``karabo.imaging.imager_factory.ImagingBackend``
- ``karabo.imaging.imager_factory.get_imager``
- ``karabo.imaging.imager_interface.ImageSpec``

Current default backend
-----------------------
If you do not explicitly choose a backend, Karabo uses ``sdp`` by default.

The default is resolved in ``parse_imaging_backend`` via:

- environment variable ``IMAGING_BACKEND`` when set
- fallback to ``ImagingBackend.SDP`` otherwise

How to select a backend
-----------------------
1. In Python code (recommended for reproducibility):

.. code-block:: python

   from karabo.imaging.imager_factory import ImagingBackend, get_imager
   from karabo.imaging.imager_interface import ImageSpec

   imager = get_imager(ImagingBackend.SDP)  # or RASCIL / WSCLEAN
   spec = ImageSpec(npix=1024, cellsize_arcsec=1.0, phase_centre_deg=(20.0, -30.0))
   dirty, psf = imager.invert(vis, spec)
   restored = imager.restore(dirty, psf)

2. Via environment variable:

.. code-block:: bash

   export IMAGING_BACKEND=sdp   # or rascil / wsclean

3. Via CLI flag in entry points that expose it (for example):

.. code-block:: bash

   python karabo/performance_test/time_karabo_reconstruction.py --imaging-backend sdp

Behavior notes
--------------
- ``ImagingBackend.SDP``:
  - ``invert`` uses the SDP imaging path.
  - ``restore`` runs SDP deconvolution + restore.

- ``ImagingBackend.RASCIL``:
  - Deprecated legacy backend. Selecting it emits a deprecation warning.
  - ``invert`` uses the RASCIL adapter path.
  - ``restore`` runs the legacy RASCIL cleaning workflow. The adapter reuses the
    visibility from the previous ``invert`` call because RASCIL cleaning does not
    consume an existing dirty/PSF FITS pair directly.

- ``ImagingBackend.WSCLEAN``:
  - ``invert`` uses the WSClean dirty imaging path and returns dirty + PSF.
  - ``restore`` runs WSClean cleaning through the common backend interface.
  - Direct WSClean classes remain available for compatibility and custom commands,
    but emit a warning. Prefer ``get_imager(ImagingBackend.WSCLEAN)``.


When to use each backend
------------------------
- Use ``sdp`` for current and preferred imaging workflows.
- Use ``wsclean`` when you want WSClean imaging through the common backend API.
- Use ``rascil`` only for legacy compatibility or result comparison.

Migration guidance
------------------
- Prefer passing backend explicitly in workflows and notebooks.
- Avoid direct imports of backend-specific imagers in user-facing notebooks.
- Keep legacy RASCIL behavior only where needed for compatibility.
