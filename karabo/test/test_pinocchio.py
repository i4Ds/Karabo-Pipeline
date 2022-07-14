"""
from karabo.simulation.pinocchio import Pinocchio

def pinocchioFun():
    p = Pinocchio()
    p.printConfig()
    p.printRedShiftRequest()
    p.runPlanner(16, 1)
    p.printPinocchioStdOutput()
    p.printPinocchioStdError()
    #p.save("/home/filip/pinocchiotest")
        
    #p.plotMassfunction()

pinocchioFun()
"""
