import enum
import os
import subprocess

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
                  f"output_directory={telescope.config_path} \n"


        test = os.listdir(telescope.config_path)

        for item in test:
            if item.endswith(".bin"):
                os.remove(os.path.join(telescope.config_path, item))


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
        np.savetxt(str(output_file_path)+'.cst', arr, delimiter=" ", header=line1 + "\n" + line2, comments='')



    def plot_beam(self,theta,phi,absdir):
        """

        :param theta: in radians
        :param phi: in radian
        :param absdir: in DBs
        :return: polar plot
        """
        fig = plt.figure()
        ax = fig.add_axes([0.1,0.1,0.8,0.8],polar=True)
        ax.pcolormesh(phi, theta, absdir) #TODO (Add check for this) X,Y & data2D must all be same dimensions
        plt.show()



