Simulation Backend Selection
============================

Overview
--------
Karabo simulation uses a unified entry point:

- ``karabo.simulation.interferometer.InterferometerSimulation.run_simulation``

The simulation backend is selected with
``karabo.simulator_backend.SimulatorBackend``.

Supported simulation backends
-----------------------------
- ``SimulatorBackend.SDP`` (value: ``"ska-sdp"``): recommended Karabo-native
  simulation path for new workflows.
- ``SimulatorBackend.OSKAR``: still supported for OSKAR-specific simulation
  workflows and telescope/beam behavior.
- ``SimulatorBackend.RASCIL``: deprecated legacy compatibility path.

For all backends, ``run_simulation(...)`` returns a Karabo ``Visibility`` wrapper.

How to select a backend
-----------------------
Use the same API and only change the ``backend`` argument:

.. code-block:: python

   from karabo.simulator_backend import SimulatorBackend
   from karabo.simulation.interferometer import InterferometerSimulation

   simulation = InterferometerSimulation()
   vis = simulation.run_simulation(
       telescope=telescope,
       sky=sky,
       observation=observation,
       backend=SimulatorBackend.SDP,  # or OSKAR / deprecated RASCIL
   )

Telescope selection note (important during transition)
------------------------------------------------------
For SDP simulations, use the SDP telescope constructor path and run the simulation
with ``backend=SimulatorBackend.SDP``. The legacy RASCIL constructor path still
works during the transition, but emits a deprecation warning.

Example:

.. code-block:: python

   from karabo.simulator_backend import SimulatorBackend
   from karabo.simulation.telescope import Telescope

   telescope = Telescope.constructor("MID", backend=SimulatorBackend.SDP)
   # ...
   vis = simulation.run_simulation(
       telescope=telescope,
       sky=sky,
       observation=observation,
       backend=SimulatorBackend.SDP,
   )

Backend behavior notes
----------------------
- OSKAR
  - Still supported.
  - Custom ``primary_beam`` passed to ``run_simulation`` is ignored.
  - Configure beam behavior through ``InterferometerSimulation`` parameters.

- SDP
  - Recommended for new Karabo-native simulation workflows.
  - Follows an MS-based simulation path in Karabo.
  - Can apply a provided custom ``primary_beam`` in simulation.

- RASCIL
  - Deprecated and kept for legacy compatibility only.
  - Follows the older MS-based simulation path and emits a deprecation warning
    when selected.

Recommendations
---------------
- Prefer ``SimulatorBackend.SDP`` in new simulation scripts/notebooks unless you
  specifically need OSKAR behavior.
- Prefer passing simulation backend explicitly in scripts/notebooks.
- Keep output format as MS for cross-backend comparability.
- Avoid ``SimulatorBackend.RASCIL`` in new workflows.
