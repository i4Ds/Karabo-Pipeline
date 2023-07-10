import os
import tempfile

import numpy as np

from karabo.simulation.line_emission import (
    gaussian_fwhm_meerkat,
    line_emission_pointing,
    plot_scatter_recon,
    simple_gaussian_beam_correction,
)
from karabo.simulation.sky_model import SkyModel


def test_line_emission_run():
    # Tests parallelised line emission simulation and beam correction
    sky_pointing = SkyModel.sky_test()
    num_sources = len(sky_pointing[:, 2])
    z_obs_pointing = np.random.uniform(0.5, 1.0, num_sources)
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, "test_line_emission")
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
