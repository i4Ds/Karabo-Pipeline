# %%
import matplotlib.pyplot as plt

from karabo.imaging.image import Image
from karabo.imaging.imager import Imager
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.sourcedetection.evaluation import SourceDetectionEvaluation
from karabo.sourcedetection.result import PyBDSFSourceDetectionResult

# Render plots inline
if __name__ == "__main__":
    ## ADD PYBDSF

    # %%
    restored = Image.read_from_file("dev/restored.fits")

    # %%
    restored.plot()

    # %%
    restored_cuts = restored.split_image(N=2, overlap=100)

    # %%
    len(restored_cuts)

    # %%
    restored_cuts[1].plot()

    # %%
    restored_cuts[2].plot()

    # %%
    from karabo.imaging.image import ImageMosaicker

    # %%
    mi = ImageMosaicker()
    test = mi.process(restored_cuts)

    # %%
    test[0].plot()

    # %%
    detection_results = PyBDSFSourceDetectionResult.detect_sources_in_image(
        restored_cuts, thresh_isl=15, thresh_pix=20
    )

    # %%
    detection_results.get_gaussian_residual_image().plot()

    # %%
    test = detection_results.get_pixel_position_of_sources()
