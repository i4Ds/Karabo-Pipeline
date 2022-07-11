"""
from karabo.simulation.pinocchio import Pinocchio

def pinocchioFun():
    p = Pinocchio()
    p.printConfig()
    p.printRedShiftRequest()
    p.run()
    p.printPinocchioStdOutput()
    p.printPinocchioStdError()
    p.save("/home/filip/pinocchiotest")
        
    #p.plotMassfunction()

pinocchioFun()
"""