Release Notes
=============

Unreleased: first RASCIL-free release
-----------------------------------------------

These notes describe the current source tree after the v0.35.0 transitional
release. They do not announce a published release or completion of the final
release-validation checklist.

Removed
~~~~~~~

* RASCIL simulation and imaging implementations, backend enum members, adapters,
  direct imager/cleaner classes, and transitional deprecation helpers.
* The RASCIL dependency from the Conda environment and recipe, and RASCIL-only
  tests and examples. Legacy configuration strings now fail with migration
  guidance instead of selecting an implementation.

Supported workflows
~~~~~~~~~~~~~~~~~~~

* Simulation through ``InterferometerSimulation`` with SDP or OSKAR.
* Dirty, PSF, and CLEAN/restored imaging through ``get_imager`` with SDP or
  WSClean, including matching backend configuration passed through the factory.
* Measurement Set I/O and FITS import without RASCIL; Karabo's power-spectrum
  calculation retains its validated normalization and angular-scale convention.
* Updated common-imaging and SDP line-emission notebooks, plus controlled
  scientific regression checks in place of retired RASCIL parity/golden tests.

Corrections and configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* SDP channel images are preserved when the compatible-dirty-image helper is
  called with ``combine_across_frequencies=False``.
* SDP restoration passes fitted beam axes in degrees, fixing the extra division
  by 3600 that made the restoring beam too small.
* ``WscleanBackendConfig.weighting`` accepts ``"natural"`` or ``"uniform"``.
  Its default remains ``None``, preserving the executable's weighting default.
* ``python-casacore`` is explicit, and ``astropy>=5.1,<5.2`` records the pinned
  SDP data-model compatibility requirement. The NumPy ``<2.0`` bound remains.

The simulation default remains OSKAR; imaging defaults to SDP unless overridden
by ``IMAGING_BACKEND``. See :doc:`migration_rascil` for removed values, unit
conversions, and supported replacements, and :doc:`main_features/backend_overview`
for the backend matrix.

Validation and follow-up
~~~~~~~~~~~~~~~~~~~~~~~~

Fresh RASCIL-free installation and selected simulation, imaging, downstream, and
notebook checks have passed during removal. The final broad clean-environment
suite and external scientific user validation remain release gates.

NumPy 2, upgrades of the pinned scientific stack, and ARM64/HPC support are not
claimed by this change. They require separately scoped implementation and
validation after removal.
