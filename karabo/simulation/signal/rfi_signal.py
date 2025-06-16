""" Radio Frequency Interference (RFI) signal simulation """

from karabo.simulation.observation import Observation
from karabo.simulation.telescope import Telescope


class RFISignal:
    """Base type for RFI simulations"""

    def __init__(self, observation: Observation, site: Telescope) -> None:
        """
        Initializes the RFISignal class.

        Args:
            observation: The observation object containing the observation details.
            site: The telescope object containing the telescope details.
        """

        self.G0_mean: float = 1.0
        """Mean of the Gaussian distribution for the starting gain amplitude."""

        self.G0_std: float = 0.0
        """Standard deviation of the Gaussian distribution for the starting gain amplitude."""  # noqa: E501

        self.Gt_std_amp: float = 0.0
        """Standard deviation of the Gaussian distribution for the variability of the gain amplitude [%]."""  # noqa: E501

        self.Gt_std_phase: float = 0.0
        """Gt_std_phase: Standard deviation of the Gaussian distribution for the variability of the gain phase [deg]."""  # noqa: E501

        self.Gt_corr_amp: float = 0.0
        """Correlation time of the Gaussian process for the time variability of the gain amplitude [hr]."""  # noqa: E501

        self.Gt_corr_phase: float = 0.0
        """Correlation time of the Gaussian process for the time variability of the gain phase [hr]."""  # noqa: E501

        self.random_seed: int = 999
        """Random seed. Set it it to a predefined value to get reproducible results."""

        self.observation = observation
        self.site = site

    def plot_uv_coverage(self) -> None:
        """Plot the UV coverage of the observation."""
        raise NotImplementedError(
            "plot_uv_coverage method is not yet implemented in RFISignal base class."
        )

    def plot_rfi_separation(self) -> None:
        """Plot the RFI separation."""
        raise NotImplementedError(
            "plot_rfi_separation method is not yet implemented in RFISignal base class."
        )

    def plot_source_altitude(self) -> None:
        """Plot the source altitude."""
        raise NotImplementedError(
            "plot_source_altitude method is not yet implemented in RFISignal base class."  # noqa: E501
        )
