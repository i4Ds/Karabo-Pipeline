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
- ``SimulatorBackend.OSKAR``
- ``SimulatorBackend.RASCIL``
- ``SimulatorBackend.SDP`` (value: ``"ska-sdp"``)

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
       backend=SimulatorBackend.SDP,  # or OSKAR / RASCIL
   )

Telescope selection note (important during transition)
------------------------------------------------------
At the moment, ``Telescope.constructor(...)`` supports telescope loading for
``OSKAR`` and ``RASCIL`` labels. For SDP simulations, use the RASCIL-compatible
telescope constructor path and run the simulation with ``backend=SimulatorBackend.SDP``.

Example:

.. code-block:: python

   from karabo.simulator_backend import SimulatorBackend
   from karabo.simulation.telescope import Telescope

   telescope = Telescope.constructor("MID", backend=SimulatorBackend.RASCIL)
   # ...
   vis = simulation.run_simulation(
       telescope=telescope,
       sky=sky,
       observation=observation,
       backend=SimulatorBackend.SDP,
   )

This is a naming/adapter transition detail; simulation dispatch itself already
supports SDP.

Backend behavior notes
----------------------
- OSKAR
  - Custom ``primary_beam`` passed to ``run_simulation`` is ignored.
  - Configure beam behavior through ``InterferometerSimulation`` parameters.

- RASCIL and SDP
  - Both follow an MS-based simulation path in Karabo.
  - Both can apply a provided custom ``primary_beam`` in simulation.

Recommendations
---------------
- Prefer passing simulation backend explicitly in scripts/notebooks.
- Keep output format as MS for cross-backend comparability.
- Avoid backend-specific calls outside Karabo abstractions in user workflows.
