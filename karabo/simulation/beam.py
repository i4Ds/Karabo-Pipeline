import enum
import os
import subprocess
from typing import Callable

import eidos
import numpy as np
from eidos.create_beam import zernike_parameters
from eidos.spatial import recon_par
from katbeam import JimBeam
from matplotlib import pyplot as plt
from astropy.stats import gaussian_fwhm_to_sigma
from karabo.error import KaraboError
from karabo.simulation.telescope import Telescope
from karabo.util.FileHandle import FileHandle
from karabo.util.data_util import get_module_path_of_module
from scipy import interpolate
from astropy import units

class PolType(enum.Enum):
    X = "X"
    Y = "Y"
    XY = "XY"


class BeamPattern:
    """
    :param
    """

    def __init__(
        self,
        cst_file_path:str,
        telescope:Telescope=None,
        freq_hz:float=0,
        pol:str='XY',
        element_type_index:int=0,
        average_fractional_error_factor_increase:float=1.1,
        ignore_data_at_pole:bool=True,
        avg_frac_error:float=0.8,
        beam_method:str='Gaussian Beam'

    ) -> None:
        self.cst_file_path : str = cst_file_path
        self.telescope : Telescope = telescope
        self.freq_hz : float = freq_hz
        self.pol : str = pol
        self.element_type_index : int = element_type_index
        self.average_fractional_error_factor_increase : float = average_fractional_error_factor_increase
        self.ignore_data_at_pole : bool = ignore_data_at_pole
        self.avg_frac_error : float = avg_frac_error
        self.beam_method : str = beam_method

    def fit_elements(
        self,
        telescope:Telescope=None,
        freq_hz:float=None,
        pol:str=None,
        element_type_index:int=None,
        average_fractional_error_factor_increase:float=None,
        ignore_data_at_pole:bool=None,
        avg_frac_error:float=None,
    ) -> None:
        if telescope is not None: self.telescope : Telescope = telescope
        if not isinstance(self.telescope, Telescope): raise KaraboError(f'`telescope` is {type(self.telescope)} but must be of type `Telescope`!')
        if freq_hz is not None: self.freq_hz : float = freq_hz
        if pol is not None: self.pol : str = pol
        if element_type_index is not None: self.element_type_index : int = element_type_index
        if average_fractional_error_factor_increase is not None: self.average_fractional_error_factor_increase : float = average_fractional_error_factor_increase
        if ignore_data_at_pole is not None: self.ignore_data_at_pole : bool = ignore_data_at_pole
        if avg_frac_error is not None: self.avg_frac_error : float = avg_frac_error

        content = (
            "[General] \n"
            "app=oskar_fit_element_data \n"
            "\n"
            "[element_fit] \n"
            f"input_cst_file={self.cst_file_path} \n"
            f"frequency_hz={self.freq_hz} \n"
            f"average_fractional_error={self.avg_frac_error} \n"
            f"pol_type={self.pol} \n"
            f"average_fractional_error_factor_increase={self.average_fractional_error_factor_increase} \n"
            f"ignore_data_at_pole={self.ignore_data_at_pole} \n"
            f"element_type_index={self.element_type_index}\n"
            f"output_directory={self.telescope.path} \n"
        )

        #test = os.listdir(telescope.path)
        #for item in test:
        #    if item.endswith(".bin"):
        #        os.remove(os.path.join(telescope.path, item))

        settings_file = FileHandle()
        file = open(settings_file.path, "wt")
        file.write(content)
        file.flush()

        fit_data_process = subprocess.Popen(
            ["oskar_fit_element_data", f"{settings_file.path}"]
        )
        fit_data_process.communicate()

    def make_cst_from_arr(self, arr, output_file_path):
        """
        Takes array of dimensions (*,8), and returns a cst files
        :param arr:
        :return:  cst file with given output filename
        """
        line1 = "Theta [deg.]  Phi   [deg.]  Abs(Dir.)[dBi   ]   Horiz(Abs)[dBi   ]  Horiz(Phase)[deg.]  Vert(Abs)[dBi   ]  Vert(Phase )[deg. ]  Ax.Ratio[dB    ]  "
        line2 = "------------------------------------------------------------------------------------------------------------------------------------------------------"
        np.savetxt(
            str(output_file_path) + ".cst",
            arr,
            delimiter=" ",
            header=line1 + "\n" + line2,
            comments="",
        )

    @staticmethod
    def get_meerkat_uhfbeam(f, pol, beamextentx,beamextenty):
        """

        :param pol:
        :param beamextent:
        :return:
        """
        beam = JimBeam("MKAT-AA-UHF-JIM-2020")
        freqlist = beam.freqMHzlist
        marginx = np.linspace(-beamextentx / 2.0, beamextentx / 2.0, int(beamextentx * 2))
        marginy = np.linspace(-beamextenty / 2.0, beamextenty / 2.0, int(beamextenty * 2))
        x, y = np.meshgrid(marginx, marginy)
        freqMHz_idx = np.where(
            freqlist == freqlist.flat[np.abs(freqlist - f).argmin()]
        )[0][0]
        freqMHz = freqlist[freqMHz_idx]
        if pol == "H":
            beampixels = beam.HH(x, y, freqMHz)
        elif pol == "V":
            beampixels = beam.VV(x, y, freqMHz)
        else:
            beampixels = beam.I(x, y, freqMHz)
            pol = "I"
        return x,y,beampixels

    @staticmethod
    def get_eidos_holographic_beam(npix, ch, dia, thres, mode="AH") -> complex:
        """
        Returns beam
        """
        B = None
        if mode == "AH":
            meerkat_beam_coeff_ah = f"{get_module_path_of_module(eidos)}/data/meerkat_beam_coeffs_ah_zp_dct.npy"
            params, freqs = zernike_parameters(meerkat_beam_coeff_ah, npix, dia, thres)
            B = recon_par(params[ch, :])
        if mode == "EM":
            meerkat_beam_coeff_em = f"{get_module_path_of_module(eidos)}/data/meerkat_beam_coeffs_em_zp_dct.npy"
            params, freqs = zernike_parameters(meerkat_beam_coeff_em, npix, dia, thres)
            B = recon_par(params[ch, :])
        return B

    @staticmethod
    def show_eidos_beam(B_ah, path=None):
        f, ax = plt.subplots(2, 2)
        ax00 = ax[0, 0]
        ax01 = ax[0, 1]
        ax10 = ax[1, 0]
        ax11 = ax[1, 1]
        B_ah[np.where(np.abs(B_ah)==0)]=1+1j
        ax00.imshow(
            10 * np.log10(np.abs(B_ah[0, 0])),
            aspect="auto",
            origin="lower",
            extent=[-5, 5, -5, 5],
        )
        ax00.set_title("E$_{00}^{h}$")
        ax01.imshow(
            10 * np.log10(np.abs(B_ah[0, 1])),
            aspect="auto",
            origin="lower",
            extent=[-5, 5, -5, 5],
        )
        ax01.set_title("E$_{01}^{h}$")
        ax10.imshow(
            10 * np.log10(np.abs(B_ah[1, 0])),
            aspect="auto",
            origin="lower",
            extent=[-5, 5, -5, 5],
        )
        ax10.set_title("E$_{10}^{h}$")
        im = ax11.imshow(
            10 * np.log10(np.abs(B_ah[1, 1])),
            aspect="auto",
            origin="lower",
            extent=[-5, 5, -5, 5],
        )
        ax11.set_title("E$_{11}^{h}$")
        ax10.set_xlabel("Deg")
        ax00.set_ylabel("Deg")
        ax11.set_xlabel("Deg")
        ax10.set_ylabel("Deg")
        plt.colorbar(im)
        if path:
            plt.savefig(path)
        plt.show(block=False)
        plt.pause(1)

    @staticmethod
    def eidos_lineplot(B_ah, B_em, npix, path=None):
        f, ax = plt.subplots(2, 1)
        ax0 = ax[0]
        ax1 = ax[1]
        ax0.plot(
            np.linspace(-5, 5, npix),
            10 * np.log10(np.abs(B_ah[0, 0]))[250],
            "o-",
            label="AH",
        )
        ax0.plot(
            np.linspace(-5, 5, npix),
            10 * np.log10(np.abs(B_em[0, 0]))[250],
            "o-",
            label="EM",
        )
        ax1.plot(
            np.linspace(-5, 5, npix),
            10 * np.log10(np.abs(B_em[0, 0]))[250]
            - 10 * np.log10(np.abs(B_ah[0, 0]))[250],
            "o-",
            label="Residual",
        )
        ax1.set_xlabel("Distance from center (deg)")
        ax0.set_ylabel("Power (dB)")
        ax0.legend()
        if path:
            plt.savefig(path)
        plt.show(block=False)
        plt.pause(1)

    @staticmethod
    def show_kat_beam(beampixels, beamextent, freq, pol, path=None):
        """

        :param beamextent:
        :param freq:
        :param pol:
        :return:
        """
        plt.imshow(
            beampixels,
            extent=[-beamextent / 2, beamextent / 2, -beamextent / 2, beamextent / 2],
        )
        plt.title("%s pol beam\nfor %s at %dMHz" % (pol, "", freq))
        plt.xlabel("deg")
        plt.ylabel("deg")
        plt.colorbar()
        if path:
            plt.savefig(path)
        plt.show(block=False)
        plt.pause(1)

    def plot_beam(self, theta, phi, absdir, path=None):
        """

        :param theta: in radians
        :param phi: in radian
        :param absdir: in DBs
        :return: polar plot
        """
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        ax.pcolormesh(
            phi, theta, absdir
        )  # TODO (Add check for this) X,Y & data2D must all be same dimensions
        if path:
            plt.savefig(path)
        plt.show(block=False)
        plt.pause(1)

    def integrate(self, theta, phi, integrand):
        theta = units.Quantity(theta, unit=units.deg).to('rad')
        phi = units.Quantity(phi, unit=units.deg).to('rad')
        # very simple quadrature, assuming uniform
        # theta, phi sampling and theta major ordering
        dtheta = np.max(np.diff(theta))
        dphi = phi[1] - phi[0]
        dsa = (dtheta * dphi * np.sin(theta)).value
        return np.sum(dsa * integrand)

    def sym_gaussian(self,theta, phi, freq, diameter, fwhm_fac=1, voltage=True, power_norm=1):
        theta = units.Quantity(theta, unit=units.deg).to('rad')
        phi = units.Quantity(phi, unit=units.deg).to('rad')

        diameter = units.Quantity(diameter, unit=units.m)
        wl = units.Quantity(freq, unit=units.MHz).to('m', equivalencies=units.spectral())

        fwhm = (fwhm_fac * wl / diameter).to('rad', equivalencies=units.dimensionless_angles())
        sigma = gaussian_fwhm_to_sigma * fwhm
        power_beam = np.exp(-theta ** 2 / 2 / sigma ** 2).value

        power_beam *= power_norm / self.integrate(theta, phi, power_beam)

        if voltage:
            return power_beam ** .5
        else:
            return power_beam

    def quad_crosspol(self,theta, phi, vcopol, voltage=True, rel_power_dB=-40):
        theta = units.Quantity(theta, unit=units.deg).to('rad').value
        phi = units.Quantity(phi, unit=units.deg).to('rad').value

        voltage_beam = theta ** 2 * vcopol * np.cos(2 * phi + np.pi / 2)

        copol_power = self.integrate(theta, phi, vcopol ** 2)
        power_norm = self.integrate(theta, phi, voltage_beam ** 2)

        voltage_beam *= (10 ** (rel_power_dB / 10) * copol_power / power_norm) ** .5

        if voltage:
            return voltage_beam
        else:
            return voltage_beam ** 2

    @staticmethod
    def get_scaled_theta_phi(theta,theta_em,phi_em,beam0):
        beam_em=interp.griddata([theta_em,phi_em], beam0, (theta, phi), method='cubic')
        return beam_em

    @staticmethod
    def cart2pol(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    @staticmethod
    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    def sim_beam(
            self,
            beam_method=None,
        ):
        if beam_method is not None: self.beam_method = beam_method
        print("Computing Primary Beam from "+str(self.beam_method))
        max_theta = 20 * units.deg
        n_theta = 180
        n_phi = 360
        copol_kwargs = {'freq': 600 * units.MHz, 'diameter': 6 * units.m, 'power_norm': 1, 'voltage': True}
        crpol_kwargs = {'rel_power_dB': -40, 'voltage': True}
        # %%
        theta_range = np.linspace(0, max_theta, n_theta)
        phi_range = np.linspace(0, 360, n_phi, endpoint=False) * units.deg  # Don't double count 0 and 360
        #x_range,y_range=pol2cart(theta_range.value, phi_range.value)
        grid_th_phi = np.meshgrid(theta_range, phi_range, indexing='ij')
        theta = np.ravel(grid_th_phi[0])
        phi = np.ravel(grid_th_phi[1])
        phi_y = phi + 90 * units.deg  # y is just 90 deg azimuthal rotation in this example
        over_360 = phi_y[phi_y >= 360 * units.deg]
        over_360 = over_360 - 360 * units.deg
        # %%
        if(self.beam_method=='Gaussian Beam'):
            vcopol_x = self.sym_gaussian(theta, phi, **copol_kwargs)
            vcrpol_x = self.quad_crosspol(theta, phi, vcopol_x, **crpol_kwargs)
            vcopol_y = self.sym_gaussian(theta, phi_y, **copol_kwargs)
            vcrpol_y = self.quad_crosspol(theta, phi_y, vcopol_y, **crpol_kwargs)
        if(self.beam_method=='EIDOS_AH'):
            npix=100
            B = self.get_eidos_holographic_beam(npix, 0, 10, 20, mode="AH")
            xy = np.meshgrid(np.linspace(-5, 5, npix), np.linspace(-5, 5, npix))
            theta_ah,phi_ah=self.cart2pol(xy[0], xy[1]);phi_ah=phi_ah*180./np.pi+180
            theta_phi_ah=np.meshgrid(theta_ah, phi_ah)
            vcopol_x = interpolate.griddata((theta_ah.flatten(), phi_ah.flatten()),
                                          np.abs(B[0][0]).flatten(), (theta, phi), method='cubic',fill_value=0)
            vcrpol_x = interpolate.griddata((theta_ah.flatten(), phi_ah.flatten()),
                                          np.abs(B[0][1]).flatten(), (theta, phi), method='cubic',fill_value=0)
            #vcrpol_x_em = quad_crosspol(theta, phi, vcopol_x_em, **crpol_kwargs)
            vcopol_y = interpolate.griddata((theta_ah.flatten(), phi_ah.flatten()),
                                          np.abs(B[1][1]).flatten(), (theta, phi), method='cubic',fill_value=0)
            vcrpol_y = interpolate.griddata((theta_ah.flatten(), phi_ah.flatten()),
                                          np.abs(B[1][0]).flatten(), (theta, phi), method='cubic',fill_value=0)
        if(self.beam_method=='EIDOS_EM'):
            npix=100
            B = self.get_eidos_holographic_beam(npix, 0, 10, 20, mode="AH")
            xy = np.meshgrid(np.linspace(-5, 5, npix), np.linspace(-5, 5, npix))
            theta_em,phi_em=self.cart2pol(xy[0], xy[1]);phi_em=phi_em*180./np.pi+180
            theta_phi_em=np.meshgrid(theta_em, phi_em)
            vcopol_x = interpolate.griddata((theta_em.flatten(), phi_em.flatten()),
                                          np.abs(B[0][0]).flatten(), (theta, phi), method='cubic',fill_value=0)
            vcrpol_x = interpolate.griddata((theta_em.flatten(), phi_em.flatten()),
                                          np.abs(B[0][1]).flatten(), (theta, phi), method='cubic',fill_value=0)
            #vcrpol_x_em = quad_crosspol(theta, phi, vcopol_x_em, **crpol_kwargs)
            vcopol_y = interpolate.griddata((theta_em.flatten(), phi_em.flatten()),
                                          np.abs(B[1][1]).flatten(), (theta, phi), method='cubic',fill_value=0)
            vcrpol_y = interpolate.griddata((theta_em.flatten(), phi_em.flatten()),
                                          np.abs(B[1][0]).flatten(), (theta, phi), method='cubic',fill_value=0)
        if(self.beam_method=='KatBeam'):
            beampixel=get_meerkat_uhfbeam(f, 'H', 30, 30)
            theta_kb,phi_kb=self.cart2pol(beampixel[0], beampixel[1]);katb_H=beampixel[2];phi_kb=phi_kb*180./np.pi+180
            vcopol_x = interpolate.griddata((theta_kb.flatten(), phi_kb.flatten()),katb_H.flatten(), (theta, phi), method='cubic',fill_value=0)
            vcrpol_x = quad_crosspol(theta, phi, vcopol_x, **crpol_kwargs)
            beampixel=get_meerkat_uhfbeam(f, 'V', 30, 30)
            theta_kb=beampixel[0]+15;phi_kb=beampixel[1]+15;katb_V=beampixel[2]
            vcopol_y = interpolate.griddata((theta_kb.flatten(), phi_kb.flatten()),katb_V.flatten(), (theta, phi), method='cubic')
            vcrpol_y = quad_crosspol(theta, phi, vcopol_y, **crpol_kwargs)
        vcopol_x[np.where(theta.value>5)]=0;vcrpol_x[np.where(theta.value>5)]=0;vcopol_y[np.where(theta.value>5)]=0;vcrpol_y[np.where(theta.value>5)]=0
        data_x = np.column_stack([
            theta.value,  # Theta [deg]
            phi.value,  # Phi [deg]
            np.zeros_like(theta).value,  # Abs dir * / Unused
            np.abs(vcopol_x),  # Abs horizontal
            np.angle(vcopol_x, deg=True),  # Phase horizontal [deg]
            np.abs(vcrpol_x),  # Abs vertical
            np.angle(vcrpol_x, deg=True),  # Phase vertical [deg]
            np.zeros_like(theta).value,  # Ax. ratio * / Unused
        ])

        data_y = np.column_stack([
            theta.value,  # Theta [deg]
            phi.value,  # Phi [deg]
            np.zeros_like(theta).value,  # Abs dir * / Unused
            np.abs(vcrpol_y),  # Abs horizontal
            np.angle(vcrpol_y, deg=True),  # Phase horizontal [deg]
            np.abs(vcopol_y),  # Abs vertical
            np.angle(vcopol_y, deg=True),  # Phase vertical [deg]
            np.zeros_like(theta).value,  # Ax. ratio * / Unused
        ])
        return grid_th_phi,vcopol_x,vcopol_y,data_x,data_y

    def plot_beam(savefile):
        grid_th_phi,vcopol_x,vcopol_y,data_x,data_y = sim_beam('EIDOS_AH')
        fig = plt.figure(figsize=(9, 4))
        co_vmin, co_vmax = -1, 1
        cr_vmin, cr_vmax = -1.e-2, 1.e-2
        fig, axs = plt.subplots(2, 2, subplot_kw={'projection': 'polar'}, figsize=(8, 8))
        XX_ax, XY_ax, YX_ax, YY_ax = axs.flat
        for ax in axs.flat:
            ax.set_rticks(np.arange(0, max_theta.value, 10))
            ax.grid(False)  # For deprecation warning
        XX_ax.set_title(r'$V_{\rm XX}$')
        XY_ax.set_title(r'$V_{\rm XY}$')
        YX_ax.set_title(r'$V_{\rm YX}$')
        YY_ax.set_title(r'$V_{\rm YY}$')
        im = XX_ax.pcolormesh(grid_th_phi[1].to('rad').value, grid_th_phi[0].value, vcopol_x.reshape(grid_th_phi[0].shape),
                              vmin=co_vmin, vmax=co_vmax)
        plt.colorbar(im, ax=XX_ax, pad=0.1)
        im = XY_ax.pcolormesh(grid_th_phi[1].to('rad').value, grid_th_phi[0].value, vcrpol_x.reshape(grid_th_phi[0].shape),
                              vmin=cr_vmin, vmax=cr_vmax)
        plt.colorbar(im, ax=XY_ax, pad=0.1)

        im = YY_ax.pcolormesh(grid_th_phi[1].to('rad').value, grid_th_phi[0].value, vcopol_y.reshape(grid_th_phi[0].shape),
                              vmin=co_vmin, vmax=co_vmax)
        plt.colorbar(im, ax=YY_ax, pad=0.1)
        im = YX_ax.pcolormesh(grid_th_phi[1].to('rad').value, grid_th_phi[0].value, vcrpol_y.reshape(grid_th_phi[0].shape),
                              vmin=cr_vmin, vmax=cr_vmax)
        plt.colorbar(im, ax=YX_ax, pad=0.1)
        for ax in axs.flat:
            ax.grid(True)
        fig.tight_layout()
        plt.savefig(savefile)
        plt.close()

    def save_meerkat_cst_file(
            self,
            cstdata:np.ndarray,
        ) -> None:
        """
        Save CST file for MeerKat telescope for the custom beams
        """
        header = """Theta [deg]
        Phi [deg]
        Abs dir *
        Abs Horiz.
        Phase Horiz. [deg]
        Abs Vert.
        Phase Vert. [deg]
        Ax. ratio *""".split('\n')
        out_header = ''.join([f'{it:>20s}' for it in header])
        out_header += '\n' + '-' * len(out_header)
        np.savetxt(self.cst_file_path,  # X polarised (port 1) (Co=H)
                   X=cstdata, header=out_header,
                   fmt='%20e', comments='', delimiter='')

    def save_cst_file(
        self,
        cstdata:np.ndarray,
        telescope:Telescope=None,
    ) -> bool:
        if telescope is None and self.telescope is not None:
            telescope = self.telescope
        telescope_type = telescope.path.split('/')[-1].split('.tm')[0] # works as long as `read_OSKAR_tm_file` sets telescope.path
        success = True
        if telescope_type == 'meerkat':
            self.save_meerkat_cst_file(cstdata=cstdata)
        else:
            success = False
        return success
