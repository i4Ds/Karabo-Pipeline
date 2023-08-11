from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Optional, Tuple, Type, TypeVar
from warnings import warn

import bdsf
import numpy as np
from bdsf.image import Image as bdsf_image
from numpy.typing import NDArray

from karabo.imaging.image import Image
from karabo.imaging.imager import Imager
from karabo.karabo_resource import KaraboResource
from karabo.util.data_util import read_CSV_to_ndarray
from karabo.util.file_handler import FileHandler
from karabo.warning import KaraboWarning

T = TypeVar("T")


class SourceDetectionResult(KaraboResource):
    def __init__(
        self,
        detected_sources: NDArray[np.float_],
        source_image: Image,
    ) -> None:
        """
        Generic Source Detection Result Class.
        Inputting your Source Detection Result as an array

        +-------+----+-----+----------------+--------------+------------+-----------+
        | index | ra | dec | pos X (pixel) | pos Y (pixel) | total_flux | peak_flux |
        +=======+====+=====+================+==============+============+===========+
        | 0     | 30 | 200 | 400           | 500           | 0.345     |    0.34540 |
        +-------+----+-----+----------------+--------------+------------+-----------+

        Rows can also be left empty if the specified value is not found by your source
        detection algorithm. More rows can also be added at the end. As they are not
        used for any internal algorithm.

        :param detected_sources: detected sources in array
        :param source_image: Image, where the source detection was performed on
        """
        self.source_image = source_image
        self.detected_sources = detected_sources

    @classmethod
    def detect_sources_in_image(
        cls: Type[T],
        image: Image,
        beam: Optional[Tuple[float, float, float]] = None,
        quiet: bool = False,
        **kwargs: Any,
    ) -> Optional[T]:
        """
        Detecting sources in an image. The Source detection is implemented with
        the PyBDSF.process_image function.
        See https://www.astron.nl/citt/pybdsf/process_image.html for more information.

        :param image: Image to perform source detection on.
        :param beam: FWHM of restoring beam. Specify as (maj, min. pos angle E of N).
            None means it will try to be extracted from the Image data.
        :return: Source Detection Result containing the found sources
        """
        if beam is None:
            if image.has_beam_parameters():
                beam = (image.header["BMAJ"], image.header["BMIN"], image.header["BPA"])
            else:
                warn(
                    KaraboWarning(
                        "No beam parameter found. Source detection might fail!"
                    )
                )

        try:
            detection = bdsf.process_image(
                input=image.path,
                beam=beam,
                quiet=quiet,
                format="csv",
                **kwargs,
            )
        except RuntimeError as e:
            wmsg = "All pixels in the image are blanked."
            if str(e) == wmsg:
                # no need to create additional warnings since `bdsf`
                # already prints an according Error message
                return None
            else:
                raise e

        return cls(detection)  # type: ignore

    def write_to_file(self, path: str) -> None:
        """
        Save Source Detection Result to ZIP Archive containing the .fits source image
        and source-finding catalog.
        :param path: path to save the zip archive as.
        """
        if path.endswith(".zip"):
            path = path[0 : len(path) - 4]
        with tempfile.TemporaryDirectory() as tmpdir:
            self.source_image.write_to_file(os.path.join(tmpdir, "source_image.fits"))
            self.__save_sources_to_csv(os.path.join(tmpdir, "detected_sources.csv"))
            shutil.make_archive(path, "zip", tmpdir)

    @staticmethod
    def guess_beam_parameters(
        imager: Imager,
        method: str = "rascil_1_iter",
    ) -> Tuple[float, float, float]:
        """
        Guess the beam parameters from the image header.
        :param imager: Imager to guess the beam parameters from
        :param method: Method to use for guessing the beam parameters.
        :return: (BMAJ, BMIN, BPA)
        """
        if method == "rascil_1_iter":
            # TODO: Investigate why those parameters need to be set.
            imager.ingest_chan_per_vis = 1
            imager.ingest_vis_nchan = 16
            # Run
            _, restored, _ = imager.imaging_rascil(
                clean_niter=1,
                clean_nmajor=1,
            )
        else:
            raise NotImplementedError("Only rascil_1_iter is implemented")

        return (
            restored.header["BMAJ"],
            restored.header["BMIN"],
            restored.header["BPA"],
        )

    @staticmethod
    def read_from_file(path: str) -> SourceDetectionResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.unpack_archive(path, tmpdir)
            source_image = Image.read_from_file(
                os.path.join(tmpdir, "source_image.fits")
            )
            source_catalouge = np.loadtxt(
                os.path.join(tmpdir, "detected_sources.csv"), delimiter=","
            )
        return SourceDetectionResult(source_catalouge, source_image)

    def __save_sources_to_csv(self, filepath: str) -> None:
        """
        Save detected Sources to CSV
        :param filepath:
        :return:
        """
        np.savetxt(filepath, self.detected_sources, delimiter=",")

    def has_source_image(self) -> bool:
        """
        Check if source image is present.
        :return: True if present, False if not present
        """
        if self.source_image is not None:
            return True
        return False

    def get_source_image(self) -> Optional[Image]:
        """
        Return the source image, where the source detection was performed on.
        :return: Karabo Image or None (if not supplied)
        """
        if self.has_source_image():
            return self.source_image
        else:
            return None

    def get_pixel_position_of_sources(self) -> NDArray[np.float_]:
        x_pos = self.detected_sources[:, 3]
        y_pos = self.detected_sources[:, 4]
        result = np.vstack((np.array(x_pos), np.array(y_pos))).T
        return result


class PyBDSFSourceDetectionResult(SourceDetectionResult):
    def __init__(
        self,
        bdsf_detection: bdsf_image,
    ) -> None:
        """
        Source Detection Result Wrapper for source detection results from PyBDSF.
        The Object allows the use of all Karabo-Source Detection
        functions on PyBDSF results
        :param bdsf_detection: PyBDSF result image
        """
        self._fh_prefix = "pybdsf_sdr"
        self._fh_verbose = True
        fh = FileHandler.get_file_handler(
            obj=self, prefix=self._fh_prefix, verbose=self._fh_verbose
        )
        sources_file = os.path.join(fh.subdir, "sources.csv")
        bdsf_detection.write_catalog(
            outfile=sources_file, catalog_type="gaul", format="csv", clobber=True
        )
        bdsf_detected_sources = read_CSV_to_ndarray(sources_file)

        detected_sources = type(self).__transform_bdsf_to_reduced_result_array(
            bdsf_detected_sources=bdsf_detected_sources,
        )

        self.bdsf_detected_sources = bdsf_detected_sources
        self.bdsf_result = bdsf_detection
        source_image = self.__get_result_image("ch0")
        super().__init__(detected_sources, source_image)

    @staticmethod
    def __transform_bdsf_to_reduced_result_array(
        bdsf_detected_sources: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        if (
            len(bdsf_detected_sources.shape) == 2
            and bdsf_detected_sources.shape[1] > 14
        ):
            sources = bdsf_detected_sources[:, [0, 4, 6, 12, 14, 8, 9]]
        else:
            wmsg = (
                "Got unexpected shape of `bdsf_detected_sources` of "
                + f"{bdsf_detected_sources.shape}, expected 2-dimensional "
                + "array with sources!"
            )
            warn(KaraboWarning(wmsg))
            sources = bdsf_detected_sources
        return sources

    def __get_result_image(self, image_type: str, **kwargs: Any) -> Image:
        fh = FileHandler.get_file_handler(
            obj=self, prefix=self._fh_prefix, verbose=self._fh_verbose
        )
        outfile = os.path.join(fh.subdir, "result.fits")
        self.bdsf_result.export_image(
            outfile=outfile,
            img_format="fits",
            img_type=image_type,
            clobber=True,
            **kwargs,
        )
        image = Image(path=outfile)
        return image

    def get_RMS_map_image(self) -> Image:
        return self.__get_result_image("rms")

    def get_mean_map_image(self) -> Image:
        return self.__get_result_image("mean")

    def get_polarized_intensity_image(self) -> Image:
        return self.__get_result_image("pi")

    def get_gaussian_residual_image(self) -> Image:
        return self.__get_result_image("gaus_resid")

    def get_gaussian_model_image(self) -> Image:
        return self.__get_result_image("gaus_model")

    def get_shapelet_residual_image(self) -> Image:
        return self.__get_result_image("shap_resid")

    def get_shapelet_model_image(self) -> Image:
        return self.__get_result_image("shap_model")

    def get_major_axis_FWHM_variation_image(self) -> Image:
        return self.__get_result_image("psf_major")

    def get_minor_axis_FWHM_variation_image(self) -> Image:
        return self.__get_result_image("psf_minor")

    def get_position_angle_variation_image(self) -> Image:
        return self.__get_result_image("psf_pa")

    def get_peak_to_total_flux_variation_image(self) -> Image:
        return self.__get_result_image("psf_ratio")

    def get_peak_to_aperture_flux_variation_image(self) -> Image:
        return self.__get_result_image("psf_ratio_aper")

    def get_island_mask(self) -> Image:
        return self.__get_result_image("island_mask")
