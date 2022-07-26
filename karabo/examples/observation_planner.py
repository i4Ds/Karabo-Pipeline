import numpy as np
import pyvista
import utm
from pyvista import examples

from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from matplotlib import cm

from karabo.util.math_util import cartesian_to_ll, long_lat_to_cartesian


def run(sky: SkyModel, tel: Telescope, observation: Observation):
    plotter = pyvista.Plotter()

    globe = pyvista.Sphere(0.99, (0, 0, 0), theta_resolution=120, phi_resolution=120, start_theta=270.01, end_theta=270)
    # Initialize the texture coordinates array
    globe.active_t_coords = np.zeros((globe.points.shape[0], 2))
    #
    # Populate by manually calculating
    for i in range(globe.points.shape[0]):
        globe.active_t_coords[i] = [
            0.5 + np.arctan2(-globe.points[i, 0], globe.points[i, 1]) / (2 * np.pi),
            0.5 + np.arcsin(globe.points[i, 2]) / np.pi,
        ]

    # And let's display it with a world map
    # tex = examples.load_globe_texture()

    tex = pyvista.read_texture('./world_textures.jpg')

    globe.texture_map_to_sphere(inplace=True, prevent_seam=False)
    plotter.add_mesh(globe, texture=tex)

    plot_long_lat_lines(plotter)

    # # globe
    # #
    # # globe.texture_map_to_sphere(inplace=True, prevent_seam=False)
    # long_lat = np.array([cartesian_to_long_lat(x) for x in globe.points])
    # globe["Data"] = long_lat[:, 1]
    # # globe["pos"] = ["{long:.2f}/{lat:.2f}".format(long=cartesian_to_long_lat(x)[0], lat=cartesian_to_long_lat(x)[1]) for
    # # x in globe.points]
    # # plotter.add_point_labels(globe, "pos", point_size=1, font_size=10)
    # plotter.add_mesh(globe, show_edges=True, colormap='viridis')

    # sky
    plot_sky(plotter, sky)

    # telescope
    pos_tel = tel.get_cartesian_position()
    # tuple_pos = tuple(map(tuple, pos_tel))
    tel_mesh = pyvista.Cone(pos_tel, pos_tel, height=0.2, radius=0.1)
    plotter.add_mesh(tel_mesh, color="#50c5ae")

    # spheres = [pyvista.Sphere(0.01, sky_coord * 5, theta_resolution=5, phi_resolution=5) for sky_coord in sky_coords]
    # for i in range(len(spheres)):
    #     flux = fluxes[i]
    #     if max_flux != min_flux:
    #         color_value = (flux - min_flux) / (max_flux - min_flux)
    #     else:
    #         color_value = flux
    #     sphere = spheres[i]
    #     color = map(color_value)
    #     plotter.add_mesh(sphere, color=color)

    # warped = mesh.warp_by_scalar('Elevation')
    # surf = warped.extract_surface().triangulate()
    # surf = surf.decimate_pro(0.75)  # reduce the density of the mesh by 75%
    # surf.plot(cmap='gist_earth')
    plotter.show()


def plot_sky(plotter, sky):
    if not sky.sources:
        return
    fluxes = sky.sources[:, 2]
    sky_coords = sky.get_cartesian_sky() * 10
    pdata = pyvista.PolyData(sky_coords)
    flux_data = np.array(fluxes, dtype=float).transpose()
    points_2 = pdata.points[:, 2]
    pdata['Data'] = flux_data
    sphere = pyvista.Sphere(0.01, phi_resolution=5, theta_resolution=5)
    pc = pdata.glyph(scale=False, geom=sphere, orient=False)
    plotter.add_mesh(pc, colormap='viridis')


def plot_long_lat_lines(plotter: pyvista.Plotter):
    longs = np.linspace(-90, 90, 9, endpoint=False)
    plotting_lats = np.linspace(-180, 180, 180)

    plotting_longs = np.linspace(-90, 90, 180)
    lats = np.linspace(-180, 180, 18, endpoint=False)

    # draw lat lines
    for lat in lats:
        locally = np.repeat(lat, len(plotting_longs))
        coord = np.vstack((plotting_longs, locally)).transpose()
        carts = [long_lat_to_cartesian(row[0], row[1]) for row in coord]
        lines = pyvista.lines_from_points(carts)
        plotter.add_mesh(lines)

    # draw labels
    for lat in lats:
        locally = np.repeat(lat, len(longs))
        coord = np.vstack((longs, locally)).transpose()
        carts = [long_lat_to_cartesian(row[0], row[1]) for row in coord]
        lines = pyvista.lines_from_points(carts)
        lines["labels"] = ["{long:.2f}/{lat:.2f}".format(long=x[0], lat=x[1]) for x in coord]
        plotter.add_point_labels(lines, "labels", point_size=1, font_size=10)
        plotter.add_mesh(lines, opacity=0)

    # draw long lines
    for long in longs:
        locally = np.repeat(long, len(plotting_lats))
        coord = np.vstack((plotting_lats, locally)).transpose()
        carts = [long_lat_to_cartesian(row[1], row[0]) for row in coord]
        lines = pyvista.lines_from_points(carts)
        plotter.add_mesh(lines, color="#ff0000")


if __name__ == "__main__":
    # sky = SkyModel.get_random_poisson_disk_sky((-10, -10), (-5, -5), 0.5, 1, 0.5)
    # sky = SkyModel.get_GLEAM_Sky()
    #
    sky = SkyModel()
    tel = Telescope.get_MEERKAT_Telescope()
    simulation = InterferometerSimulation(channel_bandwidth_hz=1e6,
                                          time_average_sec=10)
    observation = Observation(100e6,
                              phase_centre_ra_deg=240,
                              phase_centre_dec_deg=-70,
                              number_of_time_steps=24,
                              frequency_increment_hz=20e6,
                              number_of_channels=64)

    run(sky, tel, observation)
