import enum
import os
import subprocess
from katbeam import JimBeam

import numpy as np
from matplotlib import pyplot as plt

from karabo.util.FileHandle import FileHandle
from eidos.create_beam import zernike_parameters
from eidos.spatial import recon_par,jones_to_mueller_all
from karabo.test import data_path


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
        line1 = 'Theta [deg.]  Phi   [deg.]  Abs(Dir.)[dBi   ]   Horiz(Abs)[dBi   ]  Horiz(Phase)[deg.]  Vert(Abs)[dBi   ]  Vert(Phase )[deg. ]  Ax.Ratio[dB    ]  '
        line2 = '------------------------------------------------------------------------------------------------------------------------------------------------------'
        np.savetxt(str(output_file_path)+'.cst', arr, delimiter=" ", header=line1 + "\n" + line2, comments='')
    @staticmethod
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

    @staticmethod
    def get_eidos_holographic_beam(npix,ch,dia,thres,mode='AH'):
        """
        Returns beam
        """
        if(mode=='AH'):
            meerkat_beam_coeff_ah=f"{data_path}/../../data/meerkat_beam_coeffs_ah_zp_dct.npy"
            params, freqs = zernike_parameters(meerkat_beam_coeff_ah, npix, dia, thres)
            B = recon_par(params[ch,:])
        if(mode=='EM'):
            meerkat_beam_coeff_em=f"{data_path}/../../data/meerkat_beam_coeffs_em_zp_dct.npy"
            params, freqs = zernike_parameters(meerkat_beam_coeff_em, npix, dia, thres)
            B = recon_par(params[ch,:])
        return B


    @staticmethod
    def show_eidos_beam(B_ah,path=None):
        f, ax = plt.subplots(2, 2);ax00 = ax[0, 0];ax01 = ax[0, 1];ax10 = ax[1, 0];ax11 = ax[1, 1]
        ax00.imshow(10 * np.log10(np.abs(B_ah[0, 0])), aspect='auto', origin='lower', extent=[-5, 5, -5, 5]);ax00.set_title('E$_{00}^{h}$')
        ax01.imshow(10 * np.log10(np.abs(B_ah[0, 1])), aspect='auto', origin='lower', extent=[-5, 5, -5, 5]);ax01.set_title('E$_{01}^{h}$')
        ax10.imshow(10 * np.log10(np.abs(B_ah[1, 0])), aspect='auto', origin='lower', extent=[-5, 5, -5, 5]);ax10.set_title('E$_{10}^{h}$')
        im = ax11.imshow(10 * np.log10(np.abs(B_ah[1, 1])), aspect='auto', origin='lower', extent=[-5, 5, -5, 5]);ax11.set_title('E$_{11}^{h}$')
        ax10.set_xlabel('Deg');ax00.set_ylabel('Deg')
        ax11.set_xlabel('Deg');ax10.set_ylabel('Deg')
        plt.colorbar(im)
        if path:
            plt.savefig(path)
        plt.show()

    @staticmethod
    def eidos_lineplot(B_ah,B_em,npix,path=None):
        f,ax = plt.subplots(2,1);ax0=ax[0];ax1=ax[1]
        ax0.plot(np.linspace(-5,5,npix),10*np.log10(np.abs(B_ah[0,0]))[250],'o-',label='AH')
        ax0.plot(np.linspace(-5,5,npix),10*np.log10(np.abs(B_em[0,0]))[250],'o-',label='EM')
        ax1.plot(np.linspace(-5,5,npix),10*np.log10(np.abs(B_em[0,0]))[250]-10*np.log10(np.abs(B_ah[0,0]))[250],'o-',label='Residual')
        ax1.set_xlabel('Distance from center (deg)');ax0.set_ylabel('Power (dB)');ax0.legend()
        if path:
            plt.savefig(path)
        plt.show()



    @staticmethod
    def show_kat_beam(beampixels, beamextent, freq, pol, path=None):
        """

        :param beamextent:
        :param freq:
        :param pol:
        :return:
        """
        plt.imshow(beampixels, extent=[-beamextent / 2, beamextent / 2, -beamextent / 2, beamextent / 2])
        plt.title('%s pol beam\nfor %s at %dMHz' % (pol, '', freq))
        plt.xlabel('deg');plt.ylabel('deg');plt.colorbar()
        if path:
            plt.savefig(path)
        plt.show()

    def plot_beam(self, theta, phi, absdir, path=None):
        """

        :param theta: in radians
        :param phi: in radian
        :param absdir: in DBs
        :return: polar plot
        """
        fig = plt.figure()
        ax = fig.add_axes([0.1,0.1,0.8,0.8],polar=True)
        ax.pcolormesh(phi, theta, absdir) #TODO (Add check for this) X,Y & data2D must all be same dimensions
        if path:
            plt.savefig(path)
        plt.show()



