karabo.imaging.imager_wsclean
=============================

Overview
------------
This package summarizes tools and functions to be used with the imager
based on the WSClean algorithm.

For normal imaging workflows, prefer the common backend interface:
``get_imager(ImagingBackend.WSCLEAN)``. The direct classes documented here remain
available for compatibility and custom WSClean commands, but emit a warning.


Common backend configuration
----------------------------

Import ``WscleanBackendConfig`` from ``karabo.imaging.imager_factory`` and pass
it as ``config=`` to ``get_imager``. The adapter requires ``wsclean`` on ``PATH``
and a previous ``invert`` call on the same instance before ``restore``. See
:doc:`imaging_backend_selection` for an example and weighting defaults.

.. list-table:: WscleanBackendConfig arguments
   :header-rows: 1
   :widths: 28 15 57

   * - Argument
     - Default
     - Meaning / WSClean option
   * - ``combine_across_frequencies``
     - ``True``
     - Combine frequencies; ``False`` remains unsupported by the WSClean adapter.
   * - ``weighting``
     - ``None``
     - ``"natural"``, ``"uniform"``, or ``"briggs"``; maps to ``-weight``.
   * - ``robustness``
     - ``None``
     - Finite Briggs robustness. Required with ``weighting="briggs"`` and invalid
       with other weighting choices; maps to ``-weight briggs <robustness>``.
   * - ``weighting_rank_filter``
     - ``None``
     - Finite, non-negative rank-filter factor; ``-weighting-rank-filter``.
   * - ``weighting_rank_filter_size``
     - ``None``
     - Positive integer window size in uv-pixels; ``-weighting-rank-filter-size``.
   * - ``taper_gaussian_arcsec``
     - ``None``
     - Finite, positive Gaussian taper FWHM in arcseconds;
       ``-taper-gaussian <value>asec``.
   * - ``clean_niter``
     - ``100``
     - Maximum CLEAN iterations; ``-niter`` (restoration only).
   * - ``clean_mgain``
     - ``0.8``
     - Major-loop gain; ``-mgain`` (restoration only).
   * - ``clean_auto_threshold``
     - ``3``
     - Automatic CLEAN stopping threshold in noise sigma;
       ``-auto-threshold`` (restoration only).

For weighting, rank filtering, and tapering, ``None`` omits the corresponding
command-line option and preserves the installed WSClean's default. Rank-filter
factor and size can be configured independently. Explicit zero values for
robustness and rank-filter factor are passed through, not treated as unset.
All floating-point options reject NaN and infinity.

The image geometry comes from ``ImageSpec``: ``npix`` supplies both dimensions
of ``-size``, and ``cellsize_arcsec`` is converted to degrees for ``-scale``.
One shared command builder applies geometry, weighting, rank filtering, and
tapering to dirty, PSF, and CLEAN generation. Keep the configuration unchanged
between ``invert`` and ``restore`` so the reused dirty image has matching weights.

For Briggs weighting, smaller robustness values give more uniform-like weights
and larger values give more natural-like weights. Gaussian tapering reduces the
contribution of long baselines. Its FWHM describes the taper's image-plane
Gaussian, not a forced final restoring-beam size. Inspect the resulting FITS beam
and WCS when comparing reconstructions. See WSClean's official documentation for
`weighting <https://wsclean.readthedocs.io/en/latest/image_weighting.html>`_,
`rank filtering <https://wsclean.readthedocs.io/en/latest/weight_rank_filter.html>`_,
and `tapering <https://wsclean.readthedocs.io/en/latest/tapering.html>`_.

Example using an existing Measurement Set wrapped in ``vis``:

.. code-block:: python

   from karabo.imaging.imager_factory import (
       ImagingBackend,
       WscleanBackendConfig,
       get_imager,
   )
   from karabo.imaging.imager_interface import ImageSpec

   imager = get_imager(
       ImagingBackend.WSCLEAN,
       config=WscleanBackendConfig(
           weighting="briggs",
           robustness=0.5,
           weighting_rank_filter=3.0,
           weighting_rank_filter_size=32,
           taper_gaussian_arcsec=8.0,
           clean_niter=500,
       ),
   )
   # Match the phase centre to the input Measurement Set.
   spec = ImageSpec(npix=1024, cellsize_arcsec=1.0, phase_centre_deg=(20.0, -30.0))
   dirty, psf = imager.invert(vis, spec)
   restored = imager.restore(dirty, psf)

.. autoclass:: karabo.imaging.backends.wsclean_backend.WscleanBackendConfig
   :members:
   :undoc-members:

Direct classes
--------------

The direct dirty imager and cleaner also accept ``weighting``, ``robustness``,
``weighting_rank_filter``, ``weighting_rank_filter_size``, and
``taper_gaussian_arcsec`` as keyword-only arguments. They use the same validation
and command builder when generating images. Their existing constructor defaults
and direct-use warnings are retained.

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
