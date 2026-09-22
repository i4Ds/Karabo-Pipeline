Simulation guide
================

Karabo simulation starts with a sky model, telescope, and observation, then
uses ``InterferometerSimulation`` to produce a visibility data product. The
result can be imaged with either supported imaging backend.

Choose a backend
----------------

Use the backend-selection guide to choose SDP or OSKAR and construct a
compatible telescope. Select the backend explicitly in reproducible workflows.

.. toctree::
   :maxdepth: 1

   /main_features/simulation_backend_selection

Next steps
----------

After simulation, image the Measurement Set with the :doc:`imaging` guide.
For class and method signatures, see the :doc:`/main_features/simulation` API
reference.
