import tempfile
from pathlib import Path

import numpy as np
import pytest

from karabo.simulation.pinocchio import Pinocchio


@pytest.mark.skip(reason="Takes too long on GitHub Runner")
def test_pinocchio_run():
    """Validate a simple PINOCCHIO run.

    Verify that PINOCCHIO can run successfully,
    and check physical outputs against values from
    previous successful runs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Pinocchio(working_dir=tmpdir)

        # Store output data for this redshift, in addition to z = 0
        p.addRedShift("0.3")

        # Load PINOCCHIO parameters from test parameter file
        pwd = Path(__file__).parent
        config = p.loadPinocchioConfig(pwd / "pinocchio_params.txt")
        p.setConfig(config)
        p.printConfig()
        p.printRedShiftRequest()

        # Configure run planner, then execute run and save output files
        p.runPlanner(
            gbPerNode=16,
            tasksPerNode=1,
        )
        p.run(mpiThreads=2)
        p.save(tmpdir)

        # Sanity check: number of halos saved at z = 0
        # This count will always be the same, as long as the random seed stays the same
        halo_masses = np.loadtxt(
            Path(tmpdir) / "pinocchio.test.plc.out", unpack=True, usecols=(8,)
        )

        assert len(halo_masses) == 45711  # Count found from previous run of PINOCCHIO

        # Physical check: verify that Mass Function is close to analytical fit
        (m, nm, fit) = np.loadtxt(
            Path(tmpdir) / "pinocchio.0.0000.test.mf.out",
            unpack=True,
            usecols=(0, 1, 5),
        )

        errors = m * nm - m * fit
        # For this test, we use a small PINOCCHIO run,
        # in which the mass function diverges for halo masses
        # above 5 * 10**14 Msun
        errors = errors[m < 5e14]

        # Threshold chosen to be a small error margin,
        # in order to verify that PINOCCHIO successfully
        # reproduces analytical fit to the halo mass function
        assert np.sum(errors**2) < 1e-6
