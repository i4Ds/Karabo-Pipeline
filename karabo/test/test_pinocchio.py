import os
from pathlib import Path

from karabo.simulation.pinocchio import Pinocchio


def test_simple_instance():
    tmpdir = str(Path("/users", "lmachado", "PINOCCHIOOUTPUT"))
    p = Pinocchio(working_dir=tmpdir)
    p.setRunName("test")
    print(p.paramsInputPath)
    exit()
    p.printConfig()
    p.printRedShiftRequest()
    p.runPlanner(16, 1)
    p.run(mpiThreads=2)

    p.save(os.path.join(tmpdir, "subdir"))
