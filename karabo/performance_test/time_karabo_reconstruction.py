from karabo.simulation.line_emission import karabo_reconstruction
from karabo.simulation.sky_model import SkyModel

if __name__ == "__main__":
    print("Loading sky model")
    sky = SkyModel.get_BATTYE_sky()
    phase_center = [21.44213503, -30.70729488]
    print("Filtering sky model")
    sky = sky.filter_by_radius_euclidean_flat_approximation(
        0,
        2,
        phase_center[0],
        phase_center[1],
    )
    print("Reconstructing sky model")
    karabo_reconstruction(
        "/scratch/snx3000/vtimmel/karabo_folder/test_recon/",
        sky=sky,
        pdf_plot=True,
        cut=3,
    )
