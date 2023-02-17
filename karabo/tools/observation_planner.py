import math

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pyvista
from astropy.coordinates import SkyCoord
from astropy.time import Time

from karabo.imaging.imager import Imager
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.util.math_util import long_lat_to_cartesian


class ObservationPlotter:
    def __init__(self, sky, tel, observation, imager=None):
        """
        WIP
        """
        self.sky = sky
        self.tel = tel
        self.obs = observation
        self.imager = imager
        self.plotter = pyvista.Plotter()

    def plot(self):

        self.plotter.add_slider_widget(
            self.__plot_time_dependent,
            [0, 24],
            value=12,
            title="Time of the Day in Minutes",
        )

        # sky
        self.__plot_sky()
        self.__plot_sky_lines()

        self.plotter.show()

    def __plot_time_dependent(self, hour):

        #  hour_str = f"{math.floor(hour):.0f}"
        daytime = Time(
            f"2022-08-26 {math.floor(hour)}:00:00.5", scale="utc", format="iso"
        )
        # daytime = Time('2006-01-15 21:24:37.5', scale='utc', location=('120d', '45d'))
        # earth
        self.__plot_earth()
        self.__plot_long_lat_lines(daytime)

        pos_tel = self.__plot_telescope()
        self.__plot_observation(pos_tel)
        self.__plot_horizon(pos_tel)

    def __plot_horizon(self, pos_tel):
        center = pos_tel
        direction = pos_tel
        horizon_plane = pyvista.Plane(center, direction, 5, 5)
        self.plotter.add_mesh(horizon_plane, color="#ff0000", opacity=0.2)

    def __plot_telescope(self) -> npt.NDArray:
        pos_tel = self.tel.get_cartesian_position()
        tel_mesh = pyvista.Cone(pos_tel, pos_tel, height=0.2, radius=0.1)
        self.plotter.add_mesh(tel_mesh, color="#50c5ae")
        return pos_tel

    def __plot_earth(self):
        earth = pyvista.Sphere(
            0.99,
            (0, 0, 0),
            theta_resolution=120,
            phi_resolution=120,
            start_theta=270.01,
            end_theta=270,
        )
        self.plotter.add_mesh(earth, name="earth")

    def __plot_observation(self, pos_tel):
        phase_centre = np.array(
            [self.obs.phase_centre_ra_deg, self.obs.phase_centre_dec_deg]
        )
        scale = 10
        if self.imager:
            halfsize = self.imager.imaging_npixel / 2
            offset = self.imager.imaging_cellsize * halfsize
            offset_ra = np.array([offset, 0])
            offset_dec = np.array([0, offset])
            topright = phase_centre + offset_ra + offset_dec
            topleft = phase_centre - offset_ra + offset_dec
            botright = phase_centre + offset_ra - offset_dec
            botleft = phase_centre - offset_ra - offset_dec
            topright_cart = (
                self.__convert_ra_dec_to_cartesian(topright[0], topright[1]) * scale
            )
            topleft_cart = (
                self.__convert_ra_dec_to_cartesian(topleft[0], topleft[1]) * scale
            )
            botright_cart = (
                self.__convert_ra_dec_to_cartesian(botright[0], botright[1]) * scale
            )
            botleft_cart = (
                self.__convert_ra_dec_to_cartesian(botleft[0], botleft[1]) * scale
            )

            topright_line = pyvista.lines_from_points([pos_tel, topright_cart])
            topleft_line = pyvista.lines_from_points([pos_tel, topleft_cart])
            botright_line = pyvista.lines_from_points([pos_tel, botright_cart])
            botleft_line = pyvista.lines_from_points([pos_tel, botleft_cart])
            outer = pyvista.lines_from_points(
                [
                    topright_cart,
                    topleft_cart,
                    botleft_cart,
                    botright_cart,
                    topright_cart,
                ]
            )
            self.plotter.add_mesh(topright_line, "#00ff00")
            self.plotter.add_mesh(topleft_line, "#00ff00")
            self.plotter.add_mesh(botright_line, "#00ff00")
            self.plotter.add_mesh(botleft_line, "#00ff00")
            self.plotter.add_mesh(outer, "#00ff00")
        look_at = self.__convert_ra_dec_to_cartesian(
            self.obs.phase_centre_ra_deg, self.obs.phase_centre_dec_deg
        )
        look_at *= scale
        view_line = pyvista.lines_from_points([pos_tel, look_at])
        self.plotter.add_mesh(view_line, color="#ffff00")

    def __plot_sky(self):
        if self.sky.sources is None:
            return
        fluxes = self.sky.sources[:, 2]
        sky_coords = self.sky.get_cartesian_sky() * 10
        pdata = pyvista.PolyData(sky_coords)
        flux_data = np.array(fluxes, dtype=float).transpose()
        pdata["Data"] = flux_data
        sphere = pyvista.Sphere(0.01, phi_resolution=5, theta_resolution=5)
        pc = pdata.glyph(scale=False, geom=sphere, orient=False)
        self.plotter.add_mesh(pc, colormap="viridis")

    def __plot_long_lat_lines(self, daytime: Time):

        longs = np.linspace(-90, 90, 9, endpoint=False)
        longs = [daytime.earth_rotation_angle(long).value * 15 for long in longs]
        plotting_lats = np.linspace(-180, 180, 180)

        plotting_longs = np.linspace(-90, 90, 180)
        plotting_longs = [
            daytime.earth_rotation_angle(long).value * 15 for long in plotting_longs
        ]
        lats = np.linspace(-180, 180, 18, endpoint=False)

        name_counter = 0
        # draw lat lines
        for lat in lats:
            locally = np.repeat(lat, len(plotting_longs))
            coord = np.vstack((plotting_longs, locally)).transpose()
            carts = [long_lat_to_cartesian(row[0], row[1]) for row in coord]
            lines = pyvista.lines_from_points(carts)
            self.plotter.add_mesh(lines, opacity=0.5, name=f"lat_lines{name_counter}")
            name_counter += 1

        name_counter = 0
        label_counter = 0
        # draw labels
        callback = SetVisibilityCallback()
        for lat in lats:
            locally = np.repeat(lat, len(longs))
            coord = np.vstack((longs, locally)).transpose()
            carts = [long_lat_to_cartesian(row[0], row[1]) for row in coord]
            lines = pyvista.lines_from_points(carts)
            lines["labels"] = [
                "lon={long:.0f} deg, lat={lat:.0f} deg".format(long=x[1], lat=x[0])
                for x in coord
            ]
            mapper = self.plotter.add_point_labels(
                lines,
                "labels",
                point_size=1,
                font_size=10,
                name=f"lat_label{label_counter}",
            )
            label_counter += 1
            mapper.SetVisibility(False)
            self.plotter.add_mesh(lines, opacity=0, name=f"lat_labels{name_counter}")
            name_counter += 1
            callback.add_actor(mapper)

        self.plotter.add_checkbox_button_widget(
            callback, position=(10, 70), value=False, color_on="red", color_off="grey"
        )

        name_counter = 0
        # draw long lines
        for long in longs:
            locally = np.repeat(long, len(plotting_lats))
            coord = np.vstack((plotting_lats, locally)).transpose()
            carts = [long_lat_to_cartesian(row[1], row[0]) for row in coord]
            lines = pyvista.lines_from_points(carts)
            self.plotter.add_mesh(
                lines, color="#ff0000", opacity=0.5, name=f"long_lines{name_counter}"
            )
            name_counter += 1

    def __plot_sky_lines(self, ra_limit=(0, 360), dec_limit=(-90, 90)):
        min_ra, max_ra = ra_limit
        min_dec, max_dec = dec_limit
        scale = 10

        ras = np.linspace(min_ra, max_ra, 18, endpoint=False)
        plotting_decs = np.linspace(min_dec, max_dec, 180)

        plotting_ras = np.linspace(min_ra, max_ra, 90)
        decs = np.linspace(min_dec, max_dec, 9, endpoint=False)

        # draw lat lines
        for dec in decs:
            locally = np.repeat(dec, len(plotting_ras))
            coord = np.vstack((plotting_ras, locally)).transpose()
            carts = (
                np.array(
                    [
                        self.__convert_ra_dec_to_cartesian(row[0], row[1])
                        for row in coord
                    ]
                )
                * scale
            )
            lines = pyvista.lines_from_points(carts)
            self.plotter.add_mesh(lines, color="#00ffff")

        callback = SetVisibilityCallback()
        # draw labels
        for dec in decs:
            locally = np.repeat(dec, len(ras))
            coord = np.vstack((ras, locally)).transpose()
            carts = (
                np.array(
                    [
                        self.__convert_ra_dec_to_cartesian(row[0], row[1])
                        for row in coord
                    ]
                )
                * scale
            )
            lines = pyvista.lines_from_points(carts)
            lines["labels"] = [
                "ra={long:.2f} deg, dec={lat:.0f} deg".format(long=x[0], lat=x[1])
                for x in coord
            ]
            mapper = self.plotter.add_point_labels(
                lines, "labels", point_size=1, font_size=10
            )
            mapper.SetVisibility(False)
            self.plotter.add_mesh(lines, opacity=0)
            callback.add_actor(mapper)

        self.plotter.add_checkbox_button_widget(
            callback, value=False, color_on="blue", color_off="grey"
        )

        # draw ra lines
        for ra in ras:
            locally = np.repeat(ra, len(plotting_decs))
            coord = np.vstack((plotting_decs, locally)).transpose()
            carts = (
                np.array(
                    [
                        self.__convert_ra_dec_to_cartesian(row[1], row[0])
                        for row in coord
                    ]
                )
                * scale
            )
            lines = pyvista.lines_from_points(carts)
            self.plotter.add_mesh(lines, color="#0000ff")

    @staticmethod
    def __convert_ra_dec_to_cartesian(ra, dec):
        coordinate = SkyCoord(ra * u.degree, dec * u.degree, frame="icrs")
        return np.array(
            [coordinate.cartesian.x, coordinate.cartesian.y, coordinate.cartesian.z]
        )


class SetVisibilityCallback:
    """Helper callback to keep a reference to the actor being modified."""

    def __init__(self):
        self.actors = []

    def add_actor(self, actor):
        self.actors.append(actor)

    def __call__(self, state):
        for actor in self.actors:
            actor.SetVisibility(state)


def main():
    # sky = SkyModel.get_random_poisson_disk_sky((-10, -10), (-5, -5), 0.5, 1, 0.5)
    sky = SkyModel.get_GLEAM_Sky()
    tel = Telescope.get_MEERKAT_Telescope()
    _ = InterferometerSimulation(channel_bandwidth_hz=1e6, time_average_sec=10)
    observation = Observation(
        100e6,
        phase_centre_ra_deg=240,
        phase_centre_dec_deg=-70,
        number_of_time_steps=24,
        frequency_increment_hz=20e6,
        number_of_channels=64,
    )

    imager = Imager(None, imaging_cellsize=0.03, imaging_npixel=512)

    ObservationPlotter(sky, tel, observation, imager).plot()


if __name__ == "__main__":
    main()
