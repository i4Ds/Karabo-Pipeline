Simulation Backend Selection
============================

Supported backends and defaults
-------------------------------

Use ``InterferometerSimulation.run_simulation(...)`` with a
``SimulatorBackend`` enum value:

* ``SimulatorBackend.SDP`` (value ``"ska-sdp"``) selects the Karabo-native
  SKA-SDP simulation path.
* ``SimulatorBackend.OSKAR`` (value ``"OSKAR"``) selects OSKAR simulation and
  its telescope and beam options.

The default is OSKAR for both ``run_simulation`` and ``Telescope.constructor``.
Pass the backend explicitly, and construct a telescope for that backend.
A regular ``Observation`` returns a Karabo ``Visibility``; an OSKAR
``ObservationParallelized`` returns a list of visibility products.

Selecting SDP
-------------

Given a Karabo ``sky`` and ``observation``:

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

SDP writes Measurement Sets and can apply a custom SDP image supplied through
``primary_beam``. Sky-model conversion uses SKA-SDP sky components internally.
The :doc:`/examples/examples` page links an end-to-end SDP line-emission notebook.

Selecting OSKAR
---------------

Use an OSKAR telescope configuration and select OSKAR in the same entry point:

.. code-block:: python

   telescope = Telescope.constructor("EXAMPLE", backend=SimulatorBackend.OSKAR)
   vis = simulation.run_simulation(
       telescope=telescope,
       sky=sky,
       observation=observation,
       backend=SimulatorBackend.OSKAR,
       visibility_format="MS",
   )

Configure OSKAR beam behavior through ``InterferometerSimulation`` constructor
parameters. A custom ``primary_beam`` passed to ``run_simulation`` is ignored
with a warning for OSKAR.

Imaging the result
------------------

Use Measurement Set output for either :doc:`imaging_backend_selection` backend.
The simulation backend does not determine which imager you must use.

For cross-simulator frequency comparisons, check the output frequency metadata:
OSKAR uses the configured start frequency for the first channel, while SDP uses
the channel centre (start frequency plus half the channel width).

For removed selectors and older telescope-construction patterns, see
:doc:`/migration_rascil`.
