"""Superimpose two or more signals."""

import numpy as np

from karabo.error import KaraboError
from karabo.simulation.signal.typing import BaseImage, Image2D, Image3D


# pylint: disable=too-few-public-methods
class Superimpose:
    """Superimpose two or more signals."""

    @classmethod
    def _combine_2_images_2d(
        cls,
        first_image_2d: Image2D,
        second_image_2d: Image2D,
    ) -> Image2D:
        """
        Combine the data of two 2D images.

        Parameters
        ----------
        first_image_data_2d : npt.NDArray[np.float_]
            First image data to be combined
        second_image_data_2d : npt.NDArray[np.float_]
            Second image data to be combined

        Returns
        -------
        npt.NDArray[np.float_]
            Combined image data
        """
        data = np.add(second_image_2d.data, first_image_2d.data)
        return Image2D(
            data=data,
            x_label=second_image_2d.x_label,
            y_label=second_image_2d.y_label,
            redshift=second_image_2d.redshift,
            box_dims=second_image_2d.box_dims,
        )

    @classmethod
    def _combine_images_3d_2d(
        cls,
        image_3d: Image3D,
        image_2d: Image2D,
    ) -> Image3D:
        """
        Combine the data of a 3D and 2D images.

        Parameters
        ----------
        image_data_3d : npt.NDArray[np.float_]
            3D image data to be combined
        image_data_2d : npt.NDArray[np.float_]
            Image data to be combined

        Returns
        -------
        npt.NDArray[np.float_]
            Combined image data
        """
        data = np.add(image_3d.data, image_2d.data)
        return Image3D(
            data=data,
            x_label=image_3d.x_label,
            y_label=image_3d.y_label,
            z_label=image_3d.z_label,
            redshift=image_3d.redshift,
            box_dims=image_3d.box_dims,
        )

    @classmethod
    def _combine_2_images_3d(
        cls,
        first_image_3d: Image3D,
        second_image_3d: Image3D,
    ) -> Image3D:
        """
        Combine the data of two 3D images.

        Parameters
        ----------
        first_image_data_3d : npt.NDArray[np.float_]
            3D image data to be combined
        second_image_data_3d : npt.NDArray[np.float_]
            3D image data to be combined

        Returns
        -------
        npt.NDArray[np.float_]
            Combined image data
        """
        data = np.add(first_image_3d.data, second_image_3d.data)
        return Image3D(
            data=data,
            x_label=first_image_3d.x_label,
            y_label=first_image_3d.y_label,
            z_label=first_image_3d.z_label,
            redshift=first_image_3d.redshift,
            box_dims=first_image_3d.box_dims,
        )

    @classmethod
    def combine(
        cls,
        *signals: BaseImage,
    ) -> BaseImage:
        """
        Superimpose two or more signals int a single signal.

        Superimposing is done by adding each signal to the previous one. To combine a
        Image3D with a Image2D, the Image2D will be added to every layer of the Image3D.

        If only one signal is passed, it gets returned without any further processing.

        Parameters
        ----------
        signals : BaseImage
            The signals that are to be combined.

        Returns
        -------
        BaseImage
            If the input includes at least one Image3D, the returntype is a Image3D,
            else Image2D.

        Raises
        ------
        KaraboError
            When an empty signal list is passed in.
        """
        if (sig_count := len(signals)) == 1:
            return signals[0]

        if sig_count == 0:
            raise KaraboError(
                "You need to pass at least one signals to superimpose them."
            )

        image_merge = signals[0]
        for idx in range(1, len(signals)):
            next_image = signals[idx]
            if isinstance(image_merge, Image2D) and isinstance(next_image, Image2D):
                image_merge = cls._combine_2_images_2d(image_merge, next_image)
            elif isinstance(image_merge, Image3D) and isinstance(next_image, Image2D):
                image_merge = cls._combine_images_3d_2d(image_merge, next_image)
            elif isinstance(image_merge, Image2D) and isinstance(next_image, Image3D):
                image_merge = cls._combine_images_3d_2d(next_image, image_merge)
            elif isinstance(image_merge, Image3D) and isinstance(next_image, Image3D):
                image_merge = cls._combine_2_images_3d(image_merge, next_image)
            else:
                raise KaraboError(f"Unknown image types {image_merge} / {next_image}")

        return image_merge
