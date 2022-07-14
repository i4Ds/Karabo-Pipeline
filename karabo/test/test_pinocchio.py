from karabo.simulation.pinocchio import Pinocchio

def pinocchioFun():
    p = Pinocchio()
    p.setRunName("plotTest")
    p.printConfig()
    #p.printRedShiftRequest()
    #p.runPlanner(16, 1)
    p.run()
    p.plotHalos()
    p.plotMassFunction()
    p.plotPastLightCone()
    p.save("/home/filip/pinocchiotest")
        
    #p.plotMassfunction()

if __name__ == '__main__':
    pinocchioFun()
