from __future__ import annotations

import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Any, List, Literal, Optional, Tuple, Type, TypeVar
from warnings import warn

import bdsf
import numpy as np
from astropy.io import fits
from bdsf.image import Image as bdsf_image
from dask import compute, delayed  # type: ignore
from numpy.typing import NDArray

from karabo.imaging.image import Image, ImageMosaicker
from karabo.imaging.imager import Imager
from karabo.util.dask import DaskHandler
from karabo.util.data_util import read_CSV_to_ndarray
from karabo.util.file_handler import FileHandler
from karabo.warning import KaraboWarning

ImageType = Literal[
    "RMS_map",
    "mean_map",
    "polarized_intensity",
    "gaussian_residual",
    "gaussian_model",
    "shapelet_residual",
    "shapelet_model",
    "major_axis_FWHM_variation",
    "minor_axis_FWHM_variation",
    "position_angle_variation",
    "peak_to_total_flux_variation",
    "peak_to_aperture_flux_variation",
    "island_mask",
    "source",
]

BeamType = Tuple[float, float, float]

PYBDSF_TOTAL_FLUX_IDX = 12

BDSFResultIdxsToUseForKarabo = [
    0,
    4,
    6,
    12,
    14,
    8,
    9,
]  # 0: Gaus_id, 4: RA, 6: DEC, 12: Total_flux, 14: Peak_flux, 8: RA_max, 9: E_RA_max
# See: https://pybdsf.readthedocs.io/en/latest/write_catalog.html#definition-of-output-columns # noqa E501


class ISourceDetectionResult(ABC):
    """SourceDetectionResult interface."""

    @property
    @abstractmethod
    def detected_sources(self) -> NDArray[np.float_]:
        ...

    @abstractmethod
    def has_source_image(self) -> bool:
        ...

    @abstractmethod
    def get_source_image(self) -> Optional[Image]:
        ...


class SourceDetectionResult(ISourceDetectionResult):
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

    @property
    def detected_sources(self) -> NDArray[np.float_]:
        return self._detected_sources

    @detected_sources.setter
    def detected_sources(self, sources: NDArray[np.float_]) -> None:
        self._detected_sources = sources

    @classmethod
    def detect_sources_in_image(
        cls: Type[_SourceDetectionResultType],
        image: Image,
        beam: Optional[BeamType] = None,
        quiet: bool = False,
        **kwargs: Any,
    ) -> Optional[_SourceDetectionResultType]:
        """
        Detect sources in an astronomical image using PyBDSF.process_image function.

        Parameters
        ----------
        cls : Type[_SourceDetectionResultType]
            The class on which this method is called.
        image : Image or List[Image]
            Image object for source detection. Can be a single image or a list of
            images.
        beam : Optional[BeamType], optional
            The Full Width Half Maximum (FWHM) of the restoring beam, given as a tuple
            (major axis, minor axis, position angle). If None, tries to extract from
            image metadata.
        quiet : bool, default False
            If True, suppresses verbose output.
        n_splits : int, default 0
            The number of parts to split the image into for processing. A value
            greater than 1 requires Dask.
        overlap : int, default 0
            The overlap between split parts of the image in pixels.
        **kwargs : Any
            Additional keyword arguments to pass to PyBDSF.process_image function.

        Returns
        -------
        Optional[List[_SourceDetectionResultType]]
            A list of detected sources, or None if all pixels in the image are blanked
            or on failure.

        Raises
        ------
        RuntimeError
            If an unexpected error occurs during the source detection process.

        Notes
        -----
        The dask client has to be created with the setting `processes=False` to avoid
        issues with PyBDSF multiprocessing. See similar issue here:
        https://stackoverflow.com/questions/51485212/multiprocessing-gives-assertionerror-daemonic-processes-are-not-allowed-to-have # noqa
        If 'n_splits' is greater than 1 and 'use_dask' is True, the image will be split
        into multiple parts and processed in parallel using Dask. Overlap can be
        specified to prevent edge artifacts.

        If the 'beam' parameter is not provided, the method will attempt to extract the
        beam parameters from the image metadata. A warning is raised if beam parameters
        are not found.

        The PyBDSF process_image function is called for source detection, which is
        particularly designed for radio astronomical images. For details on this
        function, refer to the PyBDSF documentation.
        """
        if beam is None and not isinstance(image, list):
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
            return cls(detection)  # type: ignore
        except RuntimeError as e:
            wmsg = "All pixels in the image are blanked."
            if str(e) == wmsg:
                # no need to create additional warnings since `bdsf`
                # already prints an according Error message
                return None
            else:
                raise e

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
    ) -> BeamType:
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
        return np.vstack((np.array(x_pos), np.array(y_pos))).T


_SourceDetectionResultType = TypeVar(
    "_SourceDetectionResultType", bound=SourceDetectionResult
)


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
        tmp_dir = FileHandler().get_tmp_dir(
            prefix="pybdsf-sdr-",
            purpose="pybdsf source-detection-result disk-cache",
            unique=self,
        )
        sources_file = os.path.join(tmp_dir, "sources.csv")
        bdsf_detection.write_catalog(
            outfile=sources_file, catalog_type="gaul", format="csv", clobber=True
        )
        # If no sources are written, the file is not created
        if os.path.exists(sources_file):
            bdsf_detected_sources = read_CSV_to_ndarray(sources_file)
            detected_sources = type(self).__transform_bdsf_to_reduced_result_array(
                bdsf_detected_sources=bdsf_detected_sources,
            )
        else:
            # Empty array with shape (1,7) if no sources are found
            detected_sources = np.empty((1, len(BDSFResultIdxsToUseForKarabo)))
            # Empty array with shape (1,46).
            # 46 because those are the total columns in the bdsf catalog
            # See: https://pybdsf.readthedocs.io/en/latest/write_catalog.html#definition-of-output-columns # noqa
            bdsf_detected_sources = np.empty((1, 46))

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
            sources = bdsf_detected_sources[:, BDSFResultIdxsToUseForKarabo]
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
        tmp_dir = FileHandler().get_tmp_dir(
            prefix="pybdsf-sdr-",
            purpose="pybdsf source-detection-result disk-cache",
            unique=self,
        )
        outfile = os.path.join(tmp_dir, f"{image_type}-result.fits")
        if os.path.exists(outfile):  # allow overwriting for new results
            os.remove(path=outfile)
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

    def get_island_mask_image(self) -> Image:
        return self.__get_result_image("island_mask")


class PyBDSFSourceDetectionResultList(ISourceDetectionResult):
    """
    A class to handle the detection results of sources from PyBDSF detections.

    This class calculates the pixel positions of sources in a mosaic image and
    removes sources that are closer than a specified minimum pixel distance,
    preferring sources with higher total flux.

    Parameters
    ----------
    bdsf_detection : Optional[List[PyBDSFSourceDetectionResult]
        A list of PyBDSF source detection results.

    Attributes
    ----------
    bdsf_detection : List[PyBDSFSourceDetectionResult]
        The list of detection results from PyBDSF.
    detected_sources : NDArray[np.float_]
        Array of detected sources. Calculated based on `bdsf_detection` and
        `min_pixel_distance_between_sources`. It's populated when the property
        is accessed and not beforehand.
    min_pixel_distance_between_sources : int
        The minimum number of pixels that must separate sources to be considered
        distinct. Defaults to 5. This attribute can be modified before calling
        any function that relies on it to adjust the minimum distance criterion.
    verbose : bool
        If True, prints verbose output. Defaults to False.

    Methods
    -------
    detected_sources() -> NDArray[np.float_]:
        Returns the detected sources after applying the minimum pixel distance
        criteria. Lazy loads the value when accessed.

    Notes
    -----
    The `detected_sources` property depends on `bdsf_detection` and
    `min_pixel_distance_between_sources`. It calculates the pixel positions
    based on these inputs and filters out sources that are closer than the
    minimum distance. It also assumes a single mosaic header is applicable for
    all detections for CRPIX calibration.
    """

    def __init__(
        self,
        bdsf_detection: Optional[List[PyBDSFSourceDetectionResult]] = None,
    ) -> None:
        self.min_pixel_distance_between_sources = 5
        self.verbose = False
        self.bdsf_detection = bdsf_detection

    @staticmethod
    def detect_sources_in_images(
        images: List[Image],
        beams: Optional[List[BeamType]] = None,
        quiet: bool = True,
        **kwargs: Any,
    ) -> Optional[PyBDSFSourceDetectionResultList]:
        # Check if a list of beams is provided
        if beams is None:
            warn(
                KaraboWarning(
                    "Beam was not passed, trying to extract from image metadata."
                )
            )
            beams = [
                (
                    image.header["BMAJ"],
                    image.header["BMIN"],
                    image.header["BPA"],
                )
                for image in images
            ]
        # Check if there is a dask client
        if DaskHandler.dask_client is not None:
            func = delayed(PyBDSFSourceDetectionResult.detect_sources_in_image)
        else:
            func = PyBDSFSourceDetectionResult.detect_sources_in_image
        results: List[PyBDSFSourceDetectionResult] = []
        for cutout, beam in zip(images, beams):
            result = func(
                image=cutout,
                beam=beam,
                quiet=quiet,
                **kwargs,
            )
            results.append(result)
        if DaskHandler.dask_client is not None:
            results = compute(*results, scheduler="distributed")
        # Keep only results that are not None
        results = [result for result in results if result is not None]

        if len(results) == 0:
            return None

        return PyBDSFSourceDetectionResultList(results)

    @property
    def detected_sources(self) -> NDArray[np.float_]:
        """
        Aggregate detected sources from multiple pybdsf detection instances.

        This method concatenates detected sources from all instances in the
        `bdsf_detection`attribute.
        It identifies and removes overlapping sources using pixel overlap.

        Returns
        -------
        NDArray[np.float_]
            A numpy array of detected sources after removing overlaps. The array
            structure and content depend on the format used by individual
            detection instances.

        Notes
        -----
        This method relies on `self.__get_idx_of_overlapping_sources()` to
        identify indices of overlapping sources and `self.__drop_cast_sources()`
        to remove them.
        """
        if self.bdsf_detection is None:
            raise ValueError(
                "No PyBDSF detection results found. Did you run "
                "detect_sources_in_images()?"
            )
        _detected_sources = np.concatenate(
            [x.detected_sources for x in self.bdsf_detection],
            axis=0,
        )
        to_drop = self.__get_idx_of_overlapping_sources()
        if len(to_drop) > 0:
            _detected_sources = self.__drop_cast_sources(_detected_sources, to_drop)
        return _detected_sources

    def get_RMS_map_image(self) -> Image:
        return self.__get_result_image("RMS_map")

    def get_mean_map_image(self) -> Image:
        return self.__get_result_image("mean_map")

    def get_polarized_intensity_image(self) -> Image:
        return self.__get_result_image("polarized_intensity")

    def get_gaussian_residual_image(self) -> Image:
        return self.__get_result_image("gaussian_residual")

    def get_gaussian_model_image(self) -> Image:
        return self.__get_result_image("gaussian_model")

    def get_shapelet_residual_image(self) -> Image:
        return self.__get_result_image("shapelet_residual")

    def get_shapelet_model_image(self) -> Image:
        return self.__get_result_image("shapelet_model")

    def get_major_axis_FWHM_variation_image(self) -> Image:
        return self.__get_result_image("major_axis_FWHM_variation")

    def get_minor_axis_FWHM_variation_image(self) -> Image:
        return self.__get_result_image("minor_axis_FWHM_variation")

    def get_position_angle_variation_image(self) -> Image:
        return self.__get_result_image("position_angle_variation")

    def get_peak_to_total_flux_variation_image(self) -> Image:
        return self.__get_result_image("peak_to_total_flux_variation")

    def get_peak_to_aperture_flux_variation_image(self) -> Image:
        return self.__get_result_image("peak_to_aperture_flux_variation")

    def get_island_mask(self) -> Image:
        return self.__get_result_image("island_mask")

    def get_source_image(self) -> Image:
        return self.__get_result_image("source")

    def has_source_image(self) -> bool:
        if self.bdsf_detection is None:
            raise ValueError(
                "No PyBDSF detection results found. Did you run "
                "detect_sources_in_images()?"
            )
        sources_images = [x.has_source_image() for x in self.bdsf_detection]
        return all(sources_images)

    def __get_result_image(self, image_type: ImageType) -> Image:
        if self.bdsf_detection is None:
            raise ValueError(
                "No PyBDSF detection results found. Did you run "
                "detect_sources_in_images()?"
            )
        images = [
            getattr(result, f"get_{image_type}_image")()
            for result in self.bdsf_detection
        ]
        mi = ImageMosaicker()
        if self.verbose:
            print(f"Getting {image_type} image by mosaicking.")

        return mi.mosaic(images)[0]

    def get_pixel_position_of_sources(self) -> NDArray[np.float_]:
        """
        Calculate and return corrected pixel positions of sources in a mosaic image.

        This public method calculates the pixel positions of sources in a mosaic
        image, corrects them, and removes overlapping sources based on a specified
        minimum distance and total flux criteria.

        Returns
        -------
        NDArray[np.float_]
            A NumPy array of the corrected and filtered pixel positions of sources
            in the mosaic image.
        """
        if self.bdsf_detection is None:
            raise ValueError(
                "No PyBDSF detection results found. Did you run "
                "detect_sources_in_images()?"
            )
        to_drop = self.__get_idx_of_overlapping_sources()
        combined_positions = self.__get_corrected_positions(
            [x.get_pixel_position_of_sources() for x in self.bdsf_detection]
        )
        if len(to_drop) > 0:
            combined_positions = self.__drop_cast_sources(combined_positions, to_drop)
        else:
            if self.verbose:
                print("No sources were merged.")
        return combined_positions

    def __drop_cast_sources(
        self,
        detected_sources: NDArray[np.float_],
        to_drop: List[int],
    ) -> NDArray[np.float_]:
        if self.verbose:
            print(f"Merged in total {len(to_drop)*2} sources into {len(to_drop)}")
        # Create a boolean mask to keep sources not in to_drop
        mask = np.ones(len(detected_sources), dtype=np.bool_)
        mask[np.array(to_drop)] = False
        detected_sources = detected_sources[mask]
        return detected_sources

    def __get_corrected_positions(
        self,
        xy_poss: List[NDArray[np.float_]],
    ) -> NDArray[np.float_]:
        """
        Calculate corrected positions of detected sources in a mosaic image.

        This method adjusts the positions of detected sources from individual images
        to align them within the context of a larger mosaic image. It considers the
        differences in reference pixel positions ('CRPIX') from each image's header
        relative to the mosaic's header.

        Parameters
        ----------
        xy_poss : List[NDArray[np.float_]]
            A list of NumPy arrays. Each array contains the x and y pixel positions
            of detected sources in individual images that comprise the mosaic.

        Returns
        -------
        NDArray[np.float_]
            A NumPy array containing the combined and corrected pixel positions
            of the sources in the mosaic image.
        """
        if self.bdsf_detection is None:
            raise ValueError(
                "No PyBDSF detection results found. Did you run "
                "detect_sources_in_images()?"
            )

        # Get headers
        headers: List[fits.header.Header] = []
        for result in self.bdsf_detection:
            source_image = result.get_source_image()
            if source_image is None:
                raise ValueError(
                    "Not all PyBDSF detection results have source images. "
                    "Did you run detect_sources_in_images()?"
                )
            headers.append(source_image.header)

        # Get mosaic header
        mosaic_header = self.get_source_image().header
        # Calculate delta CRPIX for each header relative to the mosaic
        header_crpixs = np.array(
            [[header["CRPIX1"], header["CRPIX2"]] for header in headers]
        )
        mosaic_crpixs = np.array([mosaic_header["CRPIX1"], mosaic_header["CRPIX2"]])
        # Apply delta to xy positions
        corrected_positions: List[NDArray[np.float_]] = []
        for xy_pos, crpix in zip(xy_poss, header_crpixs):
            delta_crpixs = crpix - mosaic_crpixs
            corrected_positions.append(
                xy_pos - delta_crpixs
            )  # Subtract delta to align with the mosaic

        # Combine all positions into one array
        return np.concatenate(corrected_positions, axis=0)

    def __get_idx_of_overlapping_sources(
        self,
    ) -> List[int]:
        """
        Identify indices of overlapping sources in a mosaic image.

        This method calculates the pixel positions of sources in a mosaic image
        and identifies sources that are closer than a specified minimum pixel
        distance. It decides which sources to drop based on their total flux,
        preferring sources with higher total flux.

        Returns
        -------
        List[int]
            A list of indices corresponding to the sources that should be dropped
            due to overlapping.
        """
        if self.bdsf_detection is None:
            raise ValueError(
                "No PyBDSF detection results found. Did you run "
                "detect_sources_in_images()?"
            )
        # Get XY pixel position for each result
        xy_poss = [
            result.get_pixel_position_of_sources() for result in self.bdsf_detection
        ]
        if not all([result.has_source_image() for result in self.bdsf_detection]):
            raise ValueError(
                "Not all PyBDSF detection results have source images. "
                "Did you run detect_sources_in_images()?"
            )

        # Combine all positions into one array
        combined_positions = self.__get_corrected_positions(xy_poss=xy_poss)
        # Get Total Flux per Source
        total_fluxes = np.concatenate(
            [
                x.bdsf_detected_sources[:, PYBDSF_TOTAL_FLUX_IDX]
                for x in self.bdsf_detection
            ],
            axis=0,
        )
        # Check if some sources overlap
        to_drop: List[int] = []
        for i in range(len(combined_positions)):
            for j in range(
                i + 1, len(combined_positions)
            ):  # Compare with subsequent sources to avoid repetition
                dist = np.linalg.norm(combined_positions[i] - combined_positions[j])
                if dist < self.min_pixel_distance_between_sources:
                    if self.verbose:
                        print(
                            f"Source {i} and {j} are being merged "
                            f"because distance between them is {dist}."
                        )
                    if total_fluxes[i] > total_fluxes[j]:
                        to_drop.append(j)
                    else:
                        to_drop.append(i)
        # Remove duplicates
        to_drop = list(set(to_drop))
        return to_drop
