import os
import tempfile

from karabo.data.external_data import DilutedBATTYESurveyDownloadObject
from karabo.simulation.line_emission import (
    gaussian_fwhm_meerkat,
    line_emission_pointing,
    plot_scatter_recon,
    simple_gaussian_beam_correction,
)
from karabo.simulation.sky_model import SkyModel
from karabo.util.dask import DaskHandler


def test_line_emission_run():
    DaskHandler.n_threads_per_worker = 1

    # Read in the sky
    survey = DilutedBATTYESurveyDownloadObject()
    catalog_path = survey.get()
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, "test_line_emission")
        ra = 20
        dec = -30
        cut = 1.0
        sky_pointing = SkyModel.sky_from_h5_with_redshift_filtered(
            path=catalog_path, ra_deg=ra, dec_deg=dec, outer_rad=3
        )
        # Simulation of line emission observation
        dirty_im, _, header_dirty, freq_mid_dirty = line_emission_pointing(
            path_outfile=outpath, sky=sky_pointing, cut=cut, img_size=1024
        )
        plot_scatter_recon(sky_pointing, dirty_im, outpath, header_dirty, cut=cut)

        # Primary beam correction
        gauss_fwhm = gaussian_fwhm_meerkat(freq_mid_dirty)
        beam_corrected, _ = simple_gaussian_beam_correction(
            outpath, dirty_im, gauss_fwhm, cut=cut, img_size=1024
        )
        plot_scatter_recon(
            sky_pointing,
            beam_corrected,
            outpath + "_GaussianBeam_Corrected",
            header_dirty,
            cut=cut,
        )
        print("Finished")
