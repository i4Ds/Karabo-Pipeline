import os
from pathlib import Path

from karabo.simulation.line_emission import karabo_reconstruction
from karabo.simulation.sky_model import SkyModel

if __name__ == "__main__":
    print("Loading sky model")
    sky = SkyModel.get_sample_simulated_catalog()
    phase_center = [21.44213503, -30.70729488]
    print("Filtering sky model")
    sky = sky.filter_by_radius_euclidean_flat_approximation(
        0,
        2,
        phase_center[0],
        phase_center[1],
    )
    print("Reconstructing sky model")
    OUTPUT_PATH = "test_recon/"
    SCRATCH_PATH = os.getenv("SCRATCH")
    if SCRATCH_PATH is not None:
        OUTPUT_PATH = str(Path(SCRATCH_PATH) / OUTPUT_PATH)

    karabo_reconstruction(
        OUTPUT_PATH,
        sky=sky,
        pdf_plot=True,
        cut=3,
    )
