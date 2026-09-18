Migrating from RASCIL
=====================

The transitional cohabitation release (v0.35.0) supported RASCIL alongside SDP,
OSKAR, and WSClean. The current source tree removes the RASCIL implementation,
selectors, warnings, and dependency. Old workflows must select a supported
backend; there is no compatibility adapter that executes RASCIL.

Replacement APIs
----------------

.. list-table:: Upgrade map
   :header-rows: 1
   :widths: 45 55

   * - Removed usage
     - Replacement
   * - ``SimulatorBackend.RASCIL``
     - ``SimulatorBackend.SDP`` in simulation and telescope construction
   * - ``ImagingBackend.RASCIL`` or ``get_imager("rascil")``
     - ``get_imager(ImagingBackend.SDP)`` or ``get_imager(ImagingBackend.WSCLEAN)``
   * - ``IMAGING_BACKEND=rascil`` / ``--imaging-backend rascil``
     - ``sdp`` or ``wsclean`` in the same setting
   * - ``RascilDirtyImager`` / ``RascilDirtyImagerConfig``
     - ``get_imager`` plus ``ImageSpec`` and ``invert``
   * - ``RascilImageCleaner`` / ``RascilImageCleanerConfig``
     - ``SdpImagerConfig`` or ``WscleanBackendConfig`` through the factory,
       followed by ``restore``
   * - ``SkyModel.from_fits_image(..., backend="rascil")``
     - ``backend=SimulatorBackend.SDP`` (also the default for this method)
   * - ``SkyModel.convert_to_backend(SimulatorBackend.RASCIL, ...)``
     - ``SimulatorBackend.SDP`` with the same frequency and channel-bandwidth
       arguments
   * - ``LineEmissionSimulation_RASCIL.ipynb``
     - ``LineEmissionSimulation.ipynb`` in ``karabo/examples``

Removed enum attributes and modules are no longer importable. Parsing a legacy
value through ``SimulatorBackend("RASCIL")``, ``get_imager("rascil")``, or the
FITS sky-model loader raises an actionable ``ValueError``. Remove imports and
warning filters for the deleted RASCIL deprecation helpers and
``karabo.util.rascil_util``; those warnings are no longer emitted.

Use the SDP telescope constructor directly
---------------------------------------------------

For an existing Karabo ``sky`` and ``observation``:

.. code-block:: python

   from karabo.simulator_backend import SimulatorBackend
   from karabo.simulation.interferometer import InterferometerSimulation
   from karabo.simulation.telescope import Telescope

   telescope = Telescope.constructor("MID", backend=SimulatorBackend.SDP)
   simulation = InterferometerSimulation()
   vis = simulation.run_simulation(
       telescope=telescope,
       sky=sky,
       observation=observation,
       backend=SimulatorBackend.SDP,
       visibility_format="MS",
   )

Do not access the old telescope configuration aliases. Use the public telescope
constructor and simulation APIs. OSKAR simulation remains supported and remains
the default when no simulator is passed.

Replace direct imaging calls
----------------------------

For a Measurement Set in ``vis``, with phase centre RA 20 degrees, Dec -30 degrees:

.. code-block:: python

   from karabo.imaging.imager_factory import (
       ImagingBackend,
       SdpImagerConfig,
       get_imager,
   )
   from karabo.imaging.imager_interface import ImageSpec

   imager = get_imager(
       ImagingBackend.SDP,
       config=SdpImagerConfig(clean_niter=500, override_cellsize=True),
   )
   spec = ImageSpec(npix=1024, cellsize_arcsec=1.0, phase_centre_deg=(20.0, -30.0))
   dirty, psf = imager.invert(vis, spec)
   restored = imager.restore(dirty, psf)
   restored.write_to_file("restored.fits", overwrite=True)

Migration is not a one-to-one rename of cleaner options. The current SDP adapter
supports Hogbom CLEAN only. Choose supported settings explicitly; use WSClean
through the same factory with ``WscleanBackendConfig`` when appropriate.
WSClean restoration requires a preceding ``invert`` on that same instance.

Check units and scientific outputs
---------------------------------------------

* Old direct-imager ``imaging_cellsize`` values are radians. Convert them to
  ``ImageSpec.cellsize_arcsec`` with ``math.degrees(value) * 3600``.
* ``ImageSpec.phase_centre_deg`` should match the Measurement Set; it does not
  rephase input visibilities. Check the output WCS.
* Set ``override_cellsize=True`` in SDP to preserve the requested pixel scale.
  Set ``combine_across_frequencies=False`` if channel images must be preserved;
  by default SDP sums channel images.
* A restoring beam uses degrees for ``bmaj``, ``bmin``, and ``bpa``. The SDP
  restoration fix removes an erroneous extra division by 3600; images produced
  before that correction should be regenerated if the restoring-beam scale
  matters to the analysis.
* Check source position, flux units, PSF width, and residuals after migration.
  Different gridders, weighting, restoring beams, and CLEAN controls can change
  images; exact numerical equivalence to the retired backend is not promised.

Installation and modernization
------------------------------

Use the updated project environment/recipe when installing this source version.
RASCIL is no longer a dependency. ``python-casacore`` is now an explicit
dependency for Karabo's Measurement Set I/O, and Astropy is constrained to
``>=5.1,<5.2`` for the pinned SDP data-model package.

Removal enables further dependency work; it does not complete it. NumPy remains
below 2, and the SDP stack remains pinned. NumPy 2, broader dependency upgrades,
and ARM64/HPC validation are separate follow-up tasks. See :doc:`release_notes`
for the unreleased status and remaining validation.
