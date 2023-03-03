import os
import unittest
from datetime import timedelta, datetime
import numpy as np
from karabo.imaging.imager import Imager
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.interferometer import InterferometerSimulation
import oskar
import matplotlib.pyplot as plt


sky = SkyModel()
sky_data = np.zeros((81,12))
a = np.arange(-32,-27.5, 0.5)
b = np.arange(18, 22.5, 0.5)
dec_arr,ra_arr=np.meshgrid(a,b)
sky_data[:,0] = ra_arr.flatten()
sky_data[:,1] = dec_arr.flatten()
sky_data[:,2] = 1

sky.add_point_sources(sky_data)
outfile = "Oskar_ApertureArray_3h5min"

# Setting tree
params = {
    "simulator": {
        "use_gpus": True
    },
    "observation" : {
        "num_channels": 1,
        "start_frequency_hz": 8e8,
        "frequency_inc_hz": 54593550624,
        "phase_centre_ra_deg": 20,
        "phase_centre_dec_deg": -30,
        "num_time_steps": 10,
        "start_time_utc": '2000-03-20 12:06:39',
        "length": "03:05:00.000"
    },
    "telescope": {
        "input_directory": "/home/rohit/karabo/karabo-pipeline/karabo/data/meerkat.tm",
        "aperture_array/array_pattern/normalise":True,
        "normalise_beams_at_phase_centre":True,
        'pol_mode': 'Full',
        "allow_station_beam_duplication": True,
        "station_type": "Aperture Array",
        "gaussian_beam/fwhm_deg": 1,
        "gaussian_beam/ref_freq_hz": 1.5e9 # Frequency corresponding to the FWHM and the diameter of the telescope
    },
    "interferometer": {
        "oskar_vis_filename": "output_jennifer.vis",
        "channel_bandwidth_hz": 10000000,
        "time_average_sec": 8,
        "ignore_w_components": True}
}

settings = oskar.SettingsTree("oskar_sim_interferometer")
settings.from_dict(params)

# Choose the numerical precision
precision = "single"
if precision == "single":
    settings["simulator/double_precision"] = False

# Creating the sky model
#sky = oskar.Sky(precision='single')

tel = oskar.Telescope(settings=settings)


# The following line depends on the mode with which we're loading the sky (explained in documentation)
np.savetxt('sky_model.txt', sky.sources[:,:3])
sky_sim=oskar.Sky.load('sky_model.txt', precision)

sim = oskar.Interferometer(settings=settings)
sim.set_sky_model(sky_sim)
sim.run()


imager = oskar.Imager()
# Here plenty of options are available that could be found in the documentation.
# uv_filter_max can be used to change the baseline length threshold
imager.set(input_file="output_jennifer.vis", output_root="output_jennifer", fov_deg=5, image_size=4096, weighting='Uniform', uv_filter_max=3000)
imager.set_vis_phase_centre(20, -30)

output = imager.run(return_images=1)
image = output["images"][0]

# Plot the scatter plot and the sky reconstruction next to each other
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

scatter = ax1.scatter(sky[:,0], sky[:,1], c=sky[:,2], vmin=0, vmax=3, s=10, cmap='jet')
ax1.set_aspect("equal")
plt.colorbar(scatter, ax=ax1, label='Flux [Jy]')
ax1.set_xlim((17, 23))
ax1.set_ylim((-33, -27))
ax1.set_xlabel('RA [deg]')
ax1.set_ylabel('DEC [deg]')
ax1.invert_xaxis()
ax1.invert_yaxis()

recon_img = ax2.imshow(image, cmap='YlGnBu', origin='lower', vmin=0, vmax=0.5) #dirty.data[0][0],
plt.colorbar(recon_img, ax=ax2, label='Flux Density [Jy]')
plt.tight_layout()
plt.show()

#dirty.plot("Title")