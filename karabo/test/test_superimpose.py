import unittest

import numpy as np

from karabo.error import KaraboError
from karabo.simulation.signal.superimpose import Superimpose
from karabo.simulation.signal.typing import Image2D, Image3D


class SuperimposeTestCase(unittest.TestCase):
    im_2d_1 = Image2D(
        data=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        x_label=[],
        y_label=[],
        redshift=0,
        box_dims=0,
    )
    im_2d_2 = Image2D(
        data=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        x_label=[],
        y_label=[],
        redshift=0,
        box_dims=0,
    )
    im_2d_3 = Image2D(
        data=np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
        x_label=[],
        y_label=[],
        redshift=0,
        box_dims=0,
    )

    data_3d_1 = np.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )
    im_3d_1 = Image3D(
        data=data_3d_1,
        x_label=[],
        y_label=[],
        z_label=[],
        redshift=0,
        box_dims=0,
    )
    data_3d_2 = np.array(
        [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        ]
    )
    im_3d_2 = Image3D(
        data=data_3d_2,
        x_label=[],
        y_label=[],
        z_label=[],
        redshift=0,
        box_dims=0,
    )
    data_3d_3 = np.array(
        [
            [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
            [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
            [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        ]
    )
    im_3d_3 = Image3D(
        data=data_3d_3,
        x_label=[],
        y_label=[],
        z_label=[],
        redshift=0,
        box_dims=0,
    )

    def test_empty(self) -> None:
        """
        Test if the KaraboError is raised when trying to combine the elements of an
        empty list.
        """
        with self.assertRaises(KaraboError):
            Superimpose.combine()

    def test_one_elem(self) -> None:
        """
        Test if the same image comes out if only on is given to the superimpose.
        """
        self.assertEqual(
            Superimpose.combine(SuperimposeTestCase.im_2d_2),
            SuperimposeTestCase.im_2d_2,
        )
        self.assertEqual(
            Superimpose.combine(SuperimposeTestCase.im_3d_2),
            SuperimposeTestCase.im_3d_2,
        )

    def test_tow_akin_elem(self) -> None:
        """
        Test if the two images, of the same dimension, are combined correctly.
        """
        result = (
            Superimpose.combine(
                SuperimposeTestCase.im_2d_1,
                SuperimposeTestCase.im_2d_2,
            ).data
            == SuperimposeTestCase.im_2d_2.data
        )
        self.assertTrue(result.all())

        result = (
            Superimpose.combine(
                SuperimposeTestCase.im_3d_1,
                SuperimposeTestCase.im_3d_2,
            ).data
            == SuperimposeTestCase.im_3d_2.data
        )
        self.assertTrue(result.all())

    def test_tow_different_elem(self) -> None:
        """
        Test if the two images, of different dimension, are combined correctly.
        """
        result = (
            Superimpose.combine(
                SuperimposeTestCase.im_2d_1,
                SuperimposeTestCase.im_3d_2,
            ).data
            == SuperimposeTestCase.im_3d_2.data
        )
        self.assertTrue(result.all())

        result = (
            Superimpose.combine(
                SuperimposeTestCase.im_3d_1,
                SuperimposeTestCase.im_2d_2,
            ).data
            == SuperimposeTestCase.im_3d_2.data
        )
        self.assertTrue(result.all())

    def test_akin_array_elem(self) -> None:
        """
        Test if the multiple images, of the same dimension, are combined correctly.
        """
        result = Superimpose.combine(
            SuperimposeTestCase.im_2d_1,
            SuperimposeTestCase.im_2d_2,
            SuperimposeTestCase.im_2d_1,
            SuperimposeTestCase.im_2d_3,
        ).data == np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
        self.assertTrue(result.all())

        result = Superimpose.combine(
            SuperimposeTestCase.im_3d_1,
            SuperimposeTestCase.im_3d_2,
            SuperimposeTestCase.im_3d_3,
            SuperimposeTestCase.im_3d_1,
            SuperimposeTestCase.im_3d_3,
        ).data == np.array(
            [
                [[3, 5, 7], [3, 5, 7], [3, 5, 7]],
                [[3, 5, 7], [3, 5, 7], [3, 5, 7]],
                [[3, 5, 7], [3, 5, 7], [3, 5, 7]],
            ]
        )
        self.assertTrue(result.all())

    def test_different_array_elem(self) -> None:
        """
        Test if the multiple images, of different dimension, are combined correctly.
        """
        result = Superimpose.combine(
            SuperimposeTestCase.im_3d_3,
            SuperimposeTestCase.im_2d_2,
            SuperimposeTestCase.im_2d_3,
            SuperimposeTestCase.im_3d_1,
        ).data == np.array(
            [
                [[3, 4, 5], [4, 5, 6], [5, 6, 7]],
                [[3, 4, 5], [4, 5, 6], [5, 6, 7]],
                [[3, 4, 5], [4, 5, 6], [5, 6, 7]],
            ]
        )
        self.assertTrue(result.all())
