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

Measurement Set compatibility
-----------------------------

MS conversion uses the official ``ska-sdp-datamodels==1.4.1`` reader and
exporter, with a small boundary in ``karabo.simulation.sdp_ms``. Imaging still
accepts one populated field/data-description block per call. Unsorted rows and
duplicate time/baseline samples are rejected rather than silently combined.

``TIME`` is the nominal integration midpoint in UTC seconds since MJD zero;
import no longer subtracts half an integration. Existing files are interpreted
according to their recorded timestamps, not guessed provenance. Simulation
exports the actual SDP sample epochs, without shifting them by half an
integration. Its existing hour-angle/transit scheduling convention is unchanged:
the requested UTC start selects a nearby transit and is not necessarily the
first generated sample time. Relabeling these samples would misdate their UVW.

The boundary preserves channel-dependent ``WEIGHT_SPECTRUM`` where defined,
falling back to row ``WEIGHT`` otherwise, and combines ``FLAG_ROW`` with
per-sample ``FLAG``. Correlations follow MS ``CORR_TYPE`` metadata. The supported
``linearFITS`` order is normalized to linear order before Stokes-I conversion;
reversed baselines exchange cross-hands as well as conjugating visibilities and
reversing UVW. Unsupported correlation orders still raise an error. This does
not add full-Stokes imaging: the processing library's Stokes-I converter still
does not support Stokes IQ/IV/IQUV inputs.

For simulation exports, Karabo completes upstream's row-weight/noise columns,
antenna offsets, observation interval bounds, and FIELD/SOURCE epoch metadata.
These corrections can change results that previously depended on dropped flags,
unit weights, or misidentified correlations. The MS input is never modified.
This is a simulation/imaging boundary, not a lossless arbitrary-MS archiver.

FITS compatibility
------------------

SDP imaging and MGCLS loading share ``import_sdp_image_from_fits()`` in
``karabo.imaging.sdp_io``. Supported 4D FITS cubes use the official SDP
datamodels importer, including files with swapped frequency/Stokes axes.
Karabo retains in-memory normalization for 2D celestial images, 3D frequency
or Stokes cubes, and non-I singleton polarization planes, which the SDP 1.4.1
importer cannot read directly. Complete clean-beam metadata remains in degrees.

Input validation and Karabo's existing polarization ordering are preserved;
the latter intentionally differs from SDP's ``fixpol=True`` permutation for
linear/circular correlations. This compatibility path is not a polarization
correction. Missing frequency axes retain the existing 100 MHz default.

``SkyModel.from_fits_image()`` still requires data that squeeze to a 2D celestial
plane and extracts finite pixels strictly above the threshold. It uses the
same reconstructed SDP celestial coordinates as before, without writing a
temporary FITS file. Source reference frequency still comes from ``CRVAL3``
(zero when absent), independently of the image importer's frequency default.

Configuration
-------------

.. autoclass:: karabo.imaging.backends.sdp_backend.SdpImagerConfig
   :members:
   :undoc-members:

Implementation reference
------------------------

.. autoclass:: karabo.imaging.backends.sdp_backend.SdpImager
   :members: invert, restore
