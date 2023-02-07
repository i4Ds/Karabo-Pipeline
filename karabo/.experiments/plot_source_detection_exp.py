import numpy as np

from karabo.Imaging.image import open_fits_image
from karabo.simulation.sky_model import SkyModel
from karabo.sourcedetection.evaluation import (
    SourceDetectionEvaluation,
    SourceDetectionEvaluationBlock,
)
from karabo.sourcedetection.source_detection import (
    evaluate_result_with_sky,
    read_detection_from_sources_file_csv,
)


def plot_result():
    flux_range = np.linspace(0.5, 5, 10)

    evals = []
    eval_mappings = []

    for flux in flux_range:
        sky = SkyModel(np.array([[20, -30, flux]]))
        sky.setup_default_wcs([20, -30])
        detection = read_detection_from_sources_file_csv(
            f"detection_{flux}.csv", f"dirty_{flux}.fits"
        )
        evaluation = evaluate_result_with_sky(
            detection, sky, 3.878509448876288e-05, 2, False
        )
        evals.append(evaluation)
        eval_mappings.append(evaluation.__map_sky_to_detection_array())

    block = SourceDetectionEvaluationBlock(evals)
    block.flatten_plot(5, 5, 0)


if __name__ == "__main__":
    plot_result()
