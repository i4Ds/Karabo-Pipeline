from karabo.simulation.telescope import Telescope

telescope = Telescope.get_OSKAR_Example_Telescope()
telescope.plot_telescope(file="example_telescope.png")
