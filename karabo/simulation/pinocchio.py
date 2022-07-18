import os
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import List, Tuple

from karabo.simulation.sky_model import SkyModel
from karabo.util.FileHandle import FileHandle

@dataclass
class PinocchioParams:
    name: str
    isFlag: bool
    active: bool
    value: str
    comment: str

@dataclass
class PinocchioConfig:
    confDict: dict[str, list[PinocchioParams]] = field(default_factory=dict)
    maxNameLength: int = 0

@dataclass
class PinocchioRedShiftRequest:
    header: str = ""
    redShifts: list[str] = field(default_factory=list)

class Pinocchio:

    PIN_EXEC_NAME           = "pinocchio"
    PIN_EXEC_RP_NAME        = f"{PIN_EXEC_NAME}_rp"
    PIN_DEFAULT_PARAMS_FILE = f"{PIN_EXEC_NAME}_params.conf"
    PIN_DEFAULT_OUTPUT_FILE = f"{PIN_EXEC_NAME}_outs.conf"
    PIN_PARAM_FILE          = "parameter_file"
    PIN_REDSHIFT_FILE       = "outputs"

    PIN_OUT_FILEENDING      = "out"

    PRMS_CMNT = "#"
    PRMS_CMNT_SPLITTER = "%"

    PIN_OUT_VAL_PADD = 3                            # padding after the name
    PIN_OUT_COMM_PADD = 12 + PIN_OUT_VAL_PADD       # padding between name and comment

    RAD_TO_DEG = (180 / np.pi)

    def __init__(self):
        """
        Creates temp directory (wd) for the pinocchio run. 
        Load default file paths for config files, load default config file
        """
        self.wd = FileHandle(is_dir = True)
        
        # get default input files
        inputFilesPath = os.environ["CONDA_PREFIX"] + "/etc/"
        self.paramsInputPath = inputFilesPath + Pinocchio.PIN_DEFAULT_PARAMS_FILE
        self.redShiftInputPath = inputFilesPath + Pinocchio.PIN_DEFAULT_OUTPUT_FILE

        self.currConfig = self.__loadPinocchioDefaultConfig__()
        self.redShiftRequest = self.__loadPinocchioDefaultRedShiftRequest__()

        self.runConfigPath = "NotSetYETCallRun*"
        self.outputFilePath = "NotSetYETCallRun*"

        self.didRun = False

    def __loadPinocchioDefaultRedShiftRequest__(self) -> PinocchioRedShiftRequest:
        # load default redshift request (outputs) file
        return self.loadPinocchioRedShiftRequest(self.redShiftInputPath) 

    def loadPinocchioRedShiftRequest(self, path: str) -> PinocchioRedShiftRequest:
        """
        load a outputs file with the redshift requests
        """
        
        redShifts = open(path)

        rsr = PinocchioRedShiftRequest()
        rsr.header = f"{Pinocchio.PRMS_CMNT} Generated redshift output request file for Pinocchio by Karabo Framework (https://github.com/i4Ds/Karabo-Pipeline)\n"

        # skip and save header
        line: str = redShifts.readline()
        while line[0] == Pinocchio.PRMS_CMNT:
            rsr.header += line
            line = redShifts.readline()
        
        rsr.header += "\n"

        # skip empty lines
        while line[0] == "\n":
            line = redShifts.readline()

        # get redshifts
        while line:
            rsr.redShifts.append(line.strip())
            line = redShifts.readline()

        return rsr

    def addRedShift(self, rs: str) -> None:
        """
        Add a redshift to the outputs file, 0.0 is default
        """

        assert len(rs.split(".")) == 2, "Input is no double value"
        self.redShiftRequest.redShifts.append(rs)

    def removeRedShift(self, rs: str) -> None:
        """
        Remove a redshift from the outputs file
        """

        assert len(rs.split(".")) == 2, "Input is no double value"
        self.redShiftRequest.redShifts.remove(rs)

    def __loadPinocchioDefaultConfig__(self) -> PinocchioConfig:
        # load default config installed with the pinocchio conda package
        return self.loadPinocchioConfig(self.paramsInputPath)     

    def loadPinocchioConfig(self, path: str) -> PinocchioConfig:
        """
        Load given pinocchio config 
        """

        configF = open(path)
        
        # remove header
        line: str = configF.readline()
        if line[0] != Pinocchio.PRMS_CMNT:
            print("input file is broken or has no header")
            return {}
        
        c: PinocchioConfig = PinocchioConfig()
        currMapName: str = ""

        while line:
            line = configF.readline()
           
            # skip empty lines
            if len(line.strip()) == 0:
                continue
            
            # header found
            if line[0] == Pinocchio.PRMS_CMNT and line[1] == " ":
                currMapName = line[2:].strip()
                c.confDict[currMapName] = [] 
                continue

            commentSplit: list[str] = line.split(Pinocchio.PRMS_CMNT_SPLITTER)
            
            # empty comment line
            if len(commentSplit) == 2 and commentSplit[0].strip() == "":
                continue

            comment: str = commentSplit[-1].strip()
            paramPart: list[str] = commentSplit[:-1]
            lineSplit: list[str] = "".join(paramPart).split()

            if len(lineSplit) == 0:
                assert False, "no lines to split"

            name: str = lineSplit[0]
            if c.maxNameLength < len(name):
                c.maxNameLength = len(name)

            # deactivated flag found
            if len(paramPart) == 2 and len(lineSplit) == 1:
                c.confDict[currMapName].append(PinocchioParams(name, True, False, "", comment))
            
            # deactivated param found
            elif len(paramPart) == 2 and len(lineSplit) == 4:
                c.confDict[currMapName].append(PinocchioParams(name, False, False, " ".join(lineSplit[1:]), comment))

            # activated flag found
            elif len(lineSplit) == 1:
                c.confDict[currMapName].append(PinocchioParams(name, True, True, "", comment))

            # param with value found
            elif len(lineSplit) == 2:
                c.confDict[currMapName].append(PinocchioParams(name, False, True, lineSplit[1], comment))

            else:
                assert False, "invalid entry"

        configF.close()
        return c

    def getConfig(self) -> PinocchioConfig:
        """
        Get pinocchio run config as a dictionary
        
        The first call returns the default dict that can be changed and set with 
        the setConfig method
        """        
        return self.currConfig

    def setConfig(self, config : PinocchioConfig) -> None:
        """
        Replace the default pinocchio config with the given config
        """
        self.currConfig = config

    def printConfig(self):
        """
        Print the current config to the console
        """
        
        k: str
        v: list[PinocchioParams]
        for (k, v) in self.currConfig.confDict.items():
            for i in v:
                desc: str = "is a flag" if i.isFlag else f"has value = {i.value}"
                status: str = "is active" if i.active else "is inactive"
                print(f"{i.name}: {desc} and {status}, comment = {i.comment}")        

    def printRedShiftRequest(self):
        """
        Print the current red shift request that gets written into the ouputs file
        """

        k: str
        for k in self.redShiftRequest.redShifts:
            print(f"Redshift active: {k}")

    def setRunName(self, name: str):
        l : list[PinocchioParams] = self.currConfig.confDict["runProperties"]
        for i in l:
           if i.name == "RunFlag":
                i.value = name

    def getRunName(self) -> str:
        l : list[PinocchioParams] = self.currConfig.confDict["runProperties"]
        for i in l:
           if i.name == "RunFlag":
                return i.value
        
        assert False, "config not available? package installation or custom loading failed"

    def run(self, printLiveOutput = True) -> None:
        """
        run pinocchio in a temp folder
        """

        assert len(self.redShiftRequest.redShifts) > 0, "all redshifts removed from outputs file - pinocchio won't calculate anything"

        self.__writeRequiredFilesToWD()

        cmd: List[str] = [Pinocchio.PIN_EXEC_NAME, self.runConfigPath]
        self.out = subprocess.run(cmd, cwd=self.wd.path, capture_output = not printLiveOutput, text=True) 

        # mark output files
        runName = self.getRunName()
        
        self.outCatalogPath: dict(str, str) = {}
        self.outMFPath: dict(str, str) = {}

        i: str
        for i in self.redShiftRequest.redShifts:
            outPrefix = f"{Pinocchio.PIN_EXEC_NAME}.{float(i):.04f}.{runName}"

            self.outCatalogPath[i] = os.path.join(self.wd.path, f"{outPrefix}.catalog.{Pinocchio.PIN_OUT_FILEENDING}")
            self.outMFPath[i] = os.path.join(self.wd.path, f"{outPrefix}.mf.{Pinocchio.PIN_OUT_FILEENDING}")

        self.outLightConePath = os.path.join(self.wd.path, f"{Pinocchio.PIN_EXEC_NAME}.{runName}.plc.{Pinocchio.PIN_OUT_FILEENDING}")

        self.didRun = True
        
    def runPlanner(self, gbPerNode: int, tasksPerNode: int) -> None:
        """
        run the pinocchio runPlanner tool to check hardware requirements for given config
        """

        self.__writeRequiredFilesToWD()

        cmd: List[str] = [Pinocchio.PIN_EXEC_RP_NAME, self.runConfigPath, f"{gbPerNode}", f"{tasksPerNode}"]
        subprocess.run(cmd, cwd=self.wd.path, text=True)

    def printPinocchioStdOutput(self) -> None:
        """
        print the std output created during run, only available if live output was disabled
        """

        if hasattr(self, "out"):
            print(self.out.stdout)

    def printPinocchioStdError(self) -> None:
        """
        print the std error created during run, only available if live output was disabled
        """

        if hasattr(self, "out"):
            print(self.out.stderr)

    def __writeRequiredFilesToWD(self) -> None:
        self.runConfigPath = self.__writeConfigToWD__()
        self.outputFilePath = self.__writeRedShiftRequestFileToWD__()

    def __writeRedShiftRequestFileToWD__(self) -> str:
        fp: str = os.path.join(self.wd.path, Pinocchio.PIN_REDSHIFT_FILE)

        with open(os.path.join(fp), "w") as temp_file:
            temp_file.write(self.redShiftRequest.header)
            k: str
            for k in self.redShiftRequest.redShifts:
                temp_file.write(k + "\n")

        return fp

    def __writeConfigToWD__(self) -> str:
        assert self.wd is not None and self.wd.isDir

        lines: list[str] = []
        # add header
        lines.append(f"{Pinocchio.PRMS_CMNT} Generated param file for Pinocchio by Karabo Framework (https://github.com/i4Ds/Karabo-Pipeline)")
        lines.append("")

        # write entries
        k: str
        v: list[PinocchioParams]
        for (k, v) in self.currConfig.confDict.items():
            # write header
            lines.append(f"{Pinocchio.PRMS_CMNT} {k}")
            
            for i in v:
                line: str = ""
                
                # write active flag
                if not i.active:
                    line += Pinocchio.PRMS_CMNT_SPLITTER + " "
                
                # write name
                line += i.name

                # add empty spaces for formatting
                padding: int = self.currConfig.maxNameLength + Pinocchio.PIN_OUT_VAL_PADD - len(line)
                line += " " * padding

                # write value
                if not i.isFlag:
                    line += i.value

                # padd again until comment
                padding = self.currConfig.maxNameLength + Pinocchio.PIN_OUT_COMM_PADD - len(line)
                line += " " * padding

                # add comment
                line += Pinocchio.PRMS_CMNT_SPLITTER + " " + i.comment
                lines.append(line)

            # add empty line between sections
            lines.append("")

        fp: str = os.path.join(self.wd.path, Pinocchio.PIN_PARAM_FILE)

        with open(os.path.join(fp), "w") as temp_file:
            temp_file.write("\n".join(lines))

        return fp

    def save(self, outDir: str) -> None:
        """
        save the run results and the config into a folder
        """
        assert os.path.isdir(outDir), "invalid directory"
        
        shutil.copytree(self.wd.path, outDir, dirs_exist_ok=True)

        # copy config
        """
        outfile = os.path.join(outDir, Pinocchio.PIN_PARAM_FILE)
        shutil.copy(self.runConfigPath, outfile)
        print(f"copied configuration file to {outfile}")
        
        outfile = os.path.join(outDir, Pinocchio.PIN_REDSHIFT_FILE)
        shutil.copy(self.outputFilePath, os.path.join(outDir, Pinocchio.PIN_REDSHIFT_FILE))
        print(f"copied outputs file to {outfile}")
        """

    def plotHalos(self, redshift: str = "0.0", save: bool = False) -> None:
        """
        plot function from pinocchio 
        """

        (x,y,z) = np.loadtxt(self.outCatalogPath[redshift], unpack=True, usecols=(5,6,7))

        plt.figure()

        index=(z<100)
        plt.plot(x[index], y[index], 'o', c='green', label='pinocchio halos')

        plt.xlim([0, 500])
        plt.ylim([0, 500])
        plt.xscale('linear')
        plt.yscale('linear')

        plt.legend(frameon=True)
        plt.title('Large-scale structure at z=0')
        plt.xlabel(r'x (Mpc/h)', fontsize=16)
        plt.ylabel(r'y (Mpc/h)', fontsize=16)

        if save:
            plt.savefig('lss.png')
        
        plt.show() 

    def plotMassFunction(self, redshift: str = "0.0", save: bool = False) -> None:
        """
        plot function from pinocchio 
        """

        (m, nm, fit) = np.loadtxt(self.outMFPath[redshift], unpack=True, usecols=(0,1,5))

        plt.figure()

        plt.plot(m, m*nm, label='pinocchio MF', ls='-', lw=3, c='green')
        plt.plot(m, m*fit, label='Watson fit', c='blue')

        plt.xlim([0.8e13, 1.e16])
        plt.ylim([1.e-8, 1.e-3])
        plt.xscale('log')
        plt.yscale('log')

        plt.legend(frameon=True)
        plt.title('Mass function at z=0')
        plt.xlabel(r'M (M$_\odot$)', fontsize=16)
        plt.ylabel(r'M n(M) (Mpc$^{-3}$)', fontsize=16)

        if save:
            plt.savefig('mf.png')
        
        plt.show()

    def plotPastLightCone(self, redshift: str = "0.0", save: bool = False) -> None:
        """ 
        plot function from pinocchio 
        
        :param redshift: redshift file that should be plotted, defaults to "0.0"
        :type redshift: str, optional
        :param save: , defaults to False
        :type save: bool, optional
        """

        (x,y,z,m)=np.loadtxt(self.outLightConePath, unpack=True, usecols=(2,3,4,8))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        index=(m>1.e14)
        ax.scatter(x[index], y[index], z[index], marker='o', c='green')

        ax.set_xlabel('x (Mpc/h)')
        ax.set_ylabel('y (Mpc/h)')
        ax.set_zlabel('z (Mpc/h)')

        plt.title('Past-light cone in comoving coordinates')

        if save:
            plt.savefig('plc.png')
        
        plt.show()

    def getSkyModel(self, near: int = 0, far: int = 100) -> SkyModel:
        assert self.didRun, "can not get sky model if run() was never called"
        return Pinocchio.getSkyModelFromFiles(self.outLightConePath, near, far)

    def getSkyModelFromFiles(path: str, near: int = 0, far: int = 100) -> SkyModel:
        """
        Create a sky model from the pinocchio simulation cone. All halos from the near to 
        far plane (euclid distance) will be translated into the RA (right ascension - [0,360] in deg) and 
        DEC (declination - [0, 90] in deg) format. 
        
        example in 1D - this function will do the same on pinocchios 3D cone:

        |               10              60          |
        (0,0,0) --------near------------far--------->
        extracted       |<------------->|

        and translated into RA DEC

        :param near: starting distance from the (0,0,0) point in [Mpc/h], default to 0
        :param far: ending distance from the (0,0,0) point in [Mpc/h], default to 100
        """
        
        # load pinocchio data
        (x,y,z)=np.loadtxt(path, unpack=True, usecols=(2,3,4))

        # calculate RA DEC
        assert x.shape == y.shape == z.shape, "x, y, z do not have the same dimensions"
        i: int
        for i in range(len(x)):
            x_pin = x[i]
            y_pin = y[i]
            z_pin = z[i]

            r_calc_pin = np.sqrt(x_pin**2 + y_pin**2 + z_pin**2) # radial distance
            theta_calc_pin = np.arccos(z_pin/r_calc_pin) # theta
            ra = np.arctan2(y_pin, x_pin) * Pinocchio.RAD_TO_DEG # RA
            dec = (np.pi/2. - theta_calc_pin) * Pinocchio.RAD_TO_DEG # DEC

        