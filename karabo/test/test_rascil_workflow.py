import os
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_datamodels.visibility import create_visibility
from ska_sdp_datamodels.configuration import (
    create_named_configuration,
)

from ska_sdp_func_python.image.deconvolution import (
    deconvolve_cube,
)


from ska_sdp_func_python.imaging import (
    invert_visibility,
    invert_ng,
)

from rascil.processing_components import create_test_image

from ska_sdp_func_python.imaging import (
    advise_wide_field,
    create_image_from_visibility,
)

results_dir = "/home/rohit/simulations/rascil_results/"
# Construct LOW core configuration
lowr3 = create_named_configuration("MID", rmax=750.0)
# We create the visibility. This just makes the uvw, time, antenna1, antenna2,
# weight columns in a table. We subsequently fill the visibility value in by
# a predict step.
times = np.zeros([1])
frequency = np.array([1e8])
channel_bandwidth = np.array([1e6])
phasecentre = SkyCoord(
    ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
)
vt = create_visibility(
    lowr3,
    times,
    frequency,
    channel_bandwidth=channel_bandwidth,
    weight=1.0,
    phasecentre=phasecentre,
    polarisation_frame=PolarisationFrame("stokesI"),
)

advice = advise_wide_field(
    vt, guard_band_image=3.0, delA=0.1, oversampling_synthesised_beam=4.0
)
cellsize = advice["cellsize"]
# Read the venerable test image, constructing a RASCIL Image
m31image = create_test_image(
    cellsize=cellsize, frequency=frequency, phasecentre=vt.phasecentre
)

model = create_image_from_visibility(vt, cellsize=cellsize, npixel=512)
dirty, sumwt = invert_visibility(vt, model, context="2d")
psf, sumwt = invert_ng(vt, model, context="2d", dopsf=True)
comp, residual = deconvolve_cube(
    dirty,
    psf,
    niter=10000,
    threshold=0.001,
    fractional_threshold=0.001,
    window_shape="quarter",
    gain=0.7,
    scales=[0, 3, 10, 30],
)


imagename = "imaging_dirty.fits"
psffilename = "psf.fits"
print(os.path.join(results_dir, imagename))
dirty.image_acc.export_to_fits(fits_file=os.path.join(results_dir, imagename))
dirty.image_acc.export_to_fits(fits_file=os.path.join(results_dir, psffilename))
