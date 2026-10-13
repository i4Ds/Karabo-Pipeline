Backend Overview
================

Simulation and imaging are separate choices. A Measurement Set produced by
either simulator can be passed to either common imaging backend:

.. code-block:: text

   Telescope + SkyModel + Observation
                   |
       InterferometerSimulation.run_simulation
             [SDP or OSKAR]
                   |
            Visibility (MS)
                   |
          get_imager(SDP or WSCLEAN)
                   |
           invert -> dirty + PSF
                   |
           restore -> CLEAN/restored image
                   |
          analysis / source detection

.. list-table:: Supported backend selectors
   :header-rows: 1
   :widths: 20 40 40

   * - Operation
     - Selector and string value
     - Entry point
   * - SDP simulation
     - ``SimulatorBackend.SDP`` (``"ska-sdp"``)
     - ``InterferometerSimulation.run_simulation``
   * - OSKAR simulation
     - ``SimulatorBackend.OSKAR`` (``"OSKAR"``)
     - ``InterferometerSimulation.run_simulation``
   * - SDP imaging
     - ``ImagingBackend.SDP`` (``"sdp"``)
     - ``get_imager`` followed by ``invert`` and ``restore``
   * - WSClean imaging
     - ``ImagingBackend.WSCLEAN`` (``"wsclean"``)
     - ``get_imager`` followed by ``invert`` and ``restore``

Simulation defaults to OSKAR. Imaging defaults to SDP unless ``IMAGING_BACKEND``
is set. Pass both selectors explicitly for reproducible workflows. Construct the
telescope for the selected simulator; the available telescope names and beam
options depend on that simulator.

Both common imaging adapters accept Measurement Sets. The separate
:doc:`oskar_imaging` utility remains available for dirty imaging, including
OSKAR visibility files, but is not a member of ``ImagingBackend`` and does not
provide CLEAN/restoration.

See :doc:`simulation_backend_selection` and :doc:`imaging_backend_selection`
for usage, and :doc:`/migration_rascil` for upgrading older workflows.
