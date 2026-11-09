karabo.simulation
=================

Overview
------------
This module contains the functionality required for an interferometer simulation. This includes defining a sky model and selecting a telescope. 

For simulation backend selection (SDP or OSKAR), see
:doc:`simulation_backend_selection`.

Classes
-------

.. autoclass:: karabo.simulation.sky_model.SkyModel
   :no-index:
   :members:
   :special-members: __init__
   :exclude-members: __update_sky_model, __convert_ra_dec_to_cartesian

.. autoclass:: karabo.simulation.observation.Observation
   :no-index:
   :members:
   :special-members: __init__
   :exclude-members: __strfdelta

.. autoclass:: karabo.simulation.observation.ObservationLong
   :no-index:
   :members:
   :special-members: __init__
   :exclude-members: __strfdelta

.. autoclass:: karabo.simulation.telescope.Telescope
   :no-index:
   :members:
   :special-members: __init__
   :exclude-members: __update_sky_model, __convert_ra_dec_to_cartesian

.. autoclass:: karabo.simulation.interferometer.InterferometerSimulation
   :no-index:
   :members:
   :special-members: __init__
   :exclude-members: __strfdelta, __run_simulation_oskar, __run_simulation_long, __get_OSKAR_settings_tree

.. autoclass:: karabo.simulation.east_north_coordinate.EastNorthCoordinate
   :no-index:
   :members:
   :special-members: __init__
   :exclude-members:

Beam package
------------
.. automodule:: karabo.simulation.beam
   :no-index:
   :members:

Line emission package
------------------------
.. automodule:: karabo.simulation.line_emission
   :no-index:
   :members:
   :show-inheritance:

.. automodule:: karabo.simulation.line_emission_helpers
   :no-index:
   :members:
   :show-inheritance:
