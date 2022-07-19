import numpy as np
from scipy.spatial.distance import cdist

from karabo.Imaging.image import Image
from karabo.simulation.sky_model import SkyModel
from karabo.sourcedetection.evaluation import SourceDetectionEvaluation
from karabo.sourcedetection.result import SourceDetectionResult, PyBDSFSourceDetectionResult
from karabo.util.data_util import read_CSV_to_ndarray


def read_detection_from_sources_file_csv(filepath: str, source_image_path: str = None) -> SourceDetectionResult:
    """
    Reads in a CSV table and saves it in the Source Detection Result.
    The format of the CSV is according to the PyBDSF definition.:
    https://www.astron.nl/citt/pybdsf/write_catalog.html#definition-of-output-columns

    Karabo creates the output from write_catalog(format='csv', catalogue_type='gaul').
    We suggest to only read in CSV that are created with Karabo (or with PyBDSF itself with the above configuration).

    This method is mainly for convenience.
    It allows that one can save the CSV with the SourceDetectionResult.save_sources_as_csv_file("./sources.csv")
    and then read it back in.
    This helps save runtime and potential wait time, when working with the output of the source detection

    :param source_image_path: (Optional), you can also read in the source image for the detection.
            If you read this back in you can use plot() function on the SkyModelToSourceDetectionMapping
    :param filepath: file of CSV sources in the format that
    :return: SourceDetectionResult
    """
    image = None
    if source_image_path is not None:
        image = Image.open_from_file(source_image_path)
    detected_sources = read_CSV_to_ndarray(filepath)
    detection = SourceDetectionResult(detected_sources=detected_sources, source_image=image)
    return detection




