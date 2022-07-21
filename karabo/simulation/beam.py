import enum
import os
import subprocess
from katbeam import JimBeam

import numpy as np
from matplotlib import pyplot as plt

from karabo.util.FileHandle import FileHandle


class PolType(enum.Enum):
    X = "X",
    Y = "Y",
    XY = "XY"


class BeamPattern:
    """
    :param
    """

    def __init__(self, cst_file_path):
        self.cst_file_path = cst_file_path

    def fit_elements(self, telescope, freq_hz=0, pol_type=PolType.XY, avg_frac_error=0.005):
        content = "[General] \n" \
                  "app=oskar_fit_element_data \n" \
                  "\n" \
                  "[element_fit] \n" \
                  f"input_cst_file={self.cst_file_path} \n" \
                  f"frequency_hz={freq_hz} \n" \
                  f"average_fractional_error={avg_frac_error} \n" \
                  f"pol_type={pol_type.value[0]} \n" \
                  f"output_directory={telescope.file} \n"

        test = os.listdir(telescope.file)

        for item in test:
            if item.endswith(".bin"):
                os.remove(os.path.join(telescope.file, item))

        settings_file = FileHandle()
        settings_file.file.write(content)
        settings_file.file.flush()

        fit_data_process = subprocess.Popen(["oskar_fit_element_data", f"{settings_file.path}"])
        fit_data_process.communicate()

    def make_cst_from_arr(self, arr, output_file_path):
        """
        Takes array of dimensions (*,8), and returns a cst files
        :param arr:
        :return:  cst file with given output filename
        """
        line1 = 'Theta [deg.]  Phi   [deg.]  Abs(Dir.)[dBi   ]   Abs(Theta)[dBi   ]  Phase(Theta)[deg.]  Abs(Phi  )[dBi   ]  Phase(Phi  )[deg.]  Ax.Ratio[dB    ]    '
        line2 = '------------------------------------------------------------------------------------------------------------------------------------------------------'
        np.savetxt(str(output_file_path) + '.cst', arr, delimiter=" ", header=line1 + "\n" + line2, comments='')

    def get_meerkat_uhfbeam(f, pol, beamextent):
        """

        :param pol:
        :param beamextent:
        :return:
        """
        beam = JimBeam('MKAT-AA-UHF-JIM-2020');
        freqlist = beam.freqMHzlist
        margin = np.linspace(-beamextent / 2., beamextent / 2., int(beamextent * 2))
        x, y = np.meshgrid(margin, margin)
        freqMHz_idx = np.where(freqlist == freqlist.flat[np.abs(freqlist - f).argmin()])[0][0]
        freqMHz = freqlist[freqMHz_idx]
        if pol == 'H':
            beampixels = beam.HH(x, y, freqMHz)
        elif pol == 'V':
            beampixels = beam.VV(x, y, freqMHz)
        else:
            beampixels = beam.I(x, y, freqMHz)
            pol = 'I'
        return beampixels

    def show_beam(beampixels, beamextent, freq, pol):
        """

        :param beamextent:
        :param freq:
        :param pol:
        :return:
        """
        plt.imshow(beampixels, extent=[-beamextent / 2, beamextent / 2, -beamextent / 2, beamextent / 2])
        plt.title('%s pol beam\nfor %s at %dMHz' % (pol, '', freq))
        plt.xlabel('deg');
        plt.ylabel('deg');
        plt.colorbar()
        plt.show()

    def plot_beam(self, theta, phi, absdir):
        """

        :param theta: in radians
        :param phi: in radian
        :param absdir: in DBs
        :return: polar plot
        """
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        ax.pcolormesh(phi, theta, absdir)  # TODO (Add check for this) X,Y & data2D must all be same dimensions
        plt.show()
