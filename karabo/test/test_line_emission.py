import os
import unittest

import numpy as np

from karabo.simulation.line_emission import (
    gaussian_fwhm_meerkat,
    line_emission_pointing,
    plot_scatter_recon,
    simple_gaussian_beam_correction,
)
from karabo.simulation.sky_model import SkyModel


class TestLineEmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # make dir for result files
        if not os.path.exists("result/lin_em"):
            os.makedirs("result/lin_em")

    def test_line_emission_run(self):
        # Tests parallelised line emission simulation and beam correction
        outpath = "restult/lin_em/test_line_emission"
        sky_pointing = SkyModel.sky_test()
        num_sources = len(sky_pointing[:, 2])
        z_obs_pointing = np.random.uniform(0.5, 1.0, num_sources)
        dirty_im, _, header_dirty, freq_mid_dirty = line_emission_pointing(
            outpath, sky_pointing, z_obs_pointing
        )
        plot_scatter_recon(sky_pointing, dirty_im, outpath, header_dirty, cut=3.0)
        gauss_fwhm = gaussian_fwhm_meerkat(freq_mid_dirty)
        beam_corrected, _ = simple_gaussian_beam_correction(
            outpath, dirty_im, gauss_fwhm
        )
        plot_scatter_recon(
            sky_pointing,
            beam_corrected,
            outpath + "_GaussianBeam_Corrected",
            header_dirty,
            cut=3.0,
        )


if __name__ == "__main__":
    # TODO: When merging pull request #451 this test should be performed
    # TestLineEmission.test_line_emission_run(TestLineEmission)
    print("")
