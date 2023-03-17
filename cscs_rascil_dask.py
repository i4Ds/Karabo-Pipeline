from karabo.imaging.imager import Imager
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.util.dask import setup_dask_for_slurm


def main():
    # Setup Dask
    client = setup_dask_for_slurm()
    # Get GLEAM Survey Sky
    phase_center = [250, -80]
    gleam_sky = SkyModel.get_GLEAM_Sky()
    gleam_sky.explore_sky(phase_center, s=0.1)

    sky = gleam_sky.filter_by_radius(0, 0.5, phase_center[0], phase_center[1])
    sky.setup_default_wcs(phase_center=phase_center)
    askap_tel = Telescope.get_ASKAP_Telescope()

    observation_settings = Observation(
        start_frequency_hz=100e6,
        phase_centre_ra_deg=phase_center[0],
        phase_centre_dec_deg=phase_center[1],
        number_of_channels=64,
        number_of_time_steps=24,
    )

    interferometer_sim = InterferometerSimulation(channel_bandwidth_hz=1e6)
    visibility_askap = interferometer_sim.run_simulation(
        askap_tel, sky, observation_settings
    )

    imaging_npixel = 2048
    imaging_cellsize = 3.878509448876288e-05

    imager_askap = Imager(
        visibility_askap,
        imaging_npixel=imaging_npixel,
        imaging_cellsize=imaging_cellsize,
    )

    # Try differnet algorithm
    # More sources
    deconvolved, restored, residual = imager_askap.imaging_rascil(
        clean_nmajor=0,
        clean_algorithm="mmclean",
        clean_scales=[0, 6, 10, 30, 60],
        clean_fractional_threshold=0.3,
        clean_threshold=0.12e-3,
        clean_nmoment=5,
        clean_psf_support=640,
        clean_restored_output="integrated",
        use_cuda=False,
        use_dask=True,
        client=client,
    )


if __name__ == "__main__":
    main()
