import os
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import List, Tuple, Dict

from traitlets import Bool

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
    confDict: Dict[str, List[PinocchioParams]] = field(default_factory=dict)
    maxNameLength: int = 0

@dataclass
class PinocchioRedShiftRequest:
    header: str = ""
    redShifts: List[str] = field(default_factory=list)

class Pinocchio:

    PIN_EXEC_MPI            = "mpirun"
    PIN_EXEC_MPI_NO_NODES   = "-np"
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

    PIN_RIGHT_ASCENSION_IDX = 0
    PIN_DECLINATION_IDX     = 1
    PIN_I_FLUS_IDX          = 2
    PIN_Q_FLUS_IDX          = 3
    PIN_U_FLUS_IDX          = 4
    PIN_V_FLUS_IDX          = 5
    PIN_REF_F_IDX           = 6

    RAD_TO_DEG = (180 / np.pi)

    def __init__(self):
        """
        Creates temp directory (wd) for the pinocchio run. 
        Load default file paths for config files, load default config files
        """
        self.wd = FileHandle(is_dir = True)
        
        # get default input files
        inputFilesPath = os.environ["CONDA_PREFIX"] + "/etc/"
        self.paramsInputPath = inputFilesPath + Pinocchio.PIN_DEFAULT_PARAMS_FILE
        self.redShiftInputPath = inputFilesPath + Pinocchio.PIN_DEFAULT_OUTPUT_FILE

        self.currConfig = self.__loadPinocchioDefaultConfig()
        self.redShiftRequest = self.__loadPinocchioDefaultRedShiftRequest()

        self.runConfigPath = "NotSetYETCallRun*"
        self.outputFilePath = "NotSetYETCallRun*"

        self.didRun = False

    def __loadPinocchioDefaultRedShiftRequest(self) -> PinocchioRedShiftRequest:
        """
        load default redshift request (outputs) file installed with the conda package

        :return: Memory representation of the outputs file
        :rtype: PinocchioRedShiftRequest
        """

        return self.loadPinocchioRedShiftRequest(self.redShiftInputPath) 

    @staticmethod
    def loadPinocchioRedShiftRequest(path: str) -> PinocchioRedShiftRequest:
        """
        load a outputs file with the redshift requests

        :param path: path to the outputs file 
        :type path: str
        :return: Memory representation of the outputs file
        :rtype: PinocchioRedShiftRequest
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
         Add a redshift to the outputs file

        :param rs: redshift that should be added, for example "2.0"
        :type rs: str
        """

        assert len(rs.split(".")) == 2, "Input is no double value"
        self.redShiftRequest.redShifts.append(rs)

    def removeRedShift(self, rs: str) -> None:
        """
        Remove a redshift from the outputs file

        :param rs: redshift that should be removed for example "1.0"
        :type rs: str
        """

        assert len(rs.split(".")) == 2, "Input is no double value"
        self.redShiftRequest.redShifts.remove(rs)

    def __loadPinocchioDefaultConfig(self) -> PinocchioConfig:
        """
        load default pinocchio config that gets installed with the conda package

        :return: _description_
        :rtype: PinocchioConfig
        """
        # load default config installed with the pinocchio conda package
        return Pinocchio.loadPinocchioConfig(self.paramsInputPath)     

    @staticmethod
    def loadPinocchioConfig(path: str) -> PinocchioConfig:
        """
        Load given pinocchio config 

        :param path: path to the config file
        :type path: str
        :return: Pinocchio config created from file
        :rtype: PinocchioConfig
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

            commentSplit: List[str] = line.split(Pinocchio.PRMS_CMNT_SPLITTER)
            
            # empty comment line
            if len(commentSplit) == 2 and commentSplit[0].strip() == "":
                continue

            comment: str = commentSplit[-1].strip()
            paramPart: List[str] = commentSplit[:-1]
            lineSplit: List[str] = "".join(paramPart).split()

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
        Get pinocchio run config as a dictionary.
        The first call returns the default dict that can be changed and set with 
        the setConfig method
        """        

        return self.currConfig

    def setConfig(self, config : PinocchioConfig) -> None:
        """
        Replace the default pinocchio config with the given config. 
        Get a config by calling the loadPinocchioConfig() method with a valid
        path to a pinocchio config file.  

        :param config: own created config in memory - format matters!
        :type config: PinocchioConfig
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
        """
        set the name of the next run called by run()

        :param name: run name
        :type name: str
        """

        l : List[PinocchioParams] = self.currConfig.confDict["runProperties"]
        for i in l:
           if i.name == "RunFlag":
                i.value = name

    def getRunName(self) -> str:
        """
        return the specified name set by setRunName()

        :return: name of the run
        :rtype: str
        """

        l : List[PinocchioParams] = self.currConfig.confDict["runProperties"]
        for i in l:
           if i.name == "RunFlag":
                return i.value
        
        assert False, "config not available? package installation or custom loading failed"

    def run(self, mpiThreads: int = 4, printLiveOutput: Bool = True) -> None:
        """
        run pinocchio in a temp folder

        :param printLiveOutput: specify if pinocchio should print the stdout and stderror to the console
                                otherwise it gets written into memory and can be retrieved by calling
                                getPinocchioStdOutput and getPinocchioStdError, defaults to True
        :type printLiveOutput: bool, optional
        """

        assert len(self.redShiftRequest.redShifts) > 0, ("all redshifts removed from outputs file" ,
                                                            " - pinocchio won't calculate anything")

        self.__writeRequiredFilesToWD()

        cmd: List[str] = [Pinocchio.PIN_EXEC_MPI, Pinocchio.PIN_EXEC_MPI_NO_NODES, str(mpiThreads), Pinocchio.PIN_EXEC_NAME, self.runConfigPath]
        self.out = subprocess.run(cmd, cwd=self.wd.path, capture_output = not printLiveOutput, text=True) 

        # mark output files
        runName = self.getRunName()
        
        self.outCatalogPath: Dict[str, str] = {}
        self.outMFPath: Dict[str, str] = {}

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

        :param gbPerNode: defines how many GByte a node has
        :type gbPerNode: int
        :param tasksPerNode: defines how many tasks run on a node
        :type tasksPerNode: int
        """

        self.__writeRequiredFilesToWD()

        cmd: List[str] = [Pinocchio.PIN_EXEC_RP_NAME, self.runConfigPath, f"{gbPerNode}", f"{tasksPerNode}"]
        subprocess.run(cmd, cwd=self.wd.path, text=True)

    def getPinocchioStdOutput(self) -> str:
        """
        get the std output created during run, only available if live output was disabled

        :return: pinocchio std output
        :rtype: str
        """

        if hasattr(self, "out"):
            return self.out.stdout

    def getPinocchioStdError(self) -> str:
        """
        get the std error created during run, only available if live output was disabled

        :return: pinocchio std error
        :rtype: str
        """
       
        if hasattr(self, "out"):
            return self.out.stderr

    def __writeRequiredFilesToWD(self) -> None:
        """
        prepare pinocchio for a run, write the needed files to the correct position
        """

        self.runConfigPath = self.__writeConfigToWD()
        self.outputFilePath = self.__writeRedShiftRequestFileToWD()

    def __writeRedShiftRequestFileToWD(self) -> str:
        """
        create a pinocchio outputs file and write it into the cwd

        :return: path of the outputs file on the filesystem
        :rtype: str
        """

        fp: str = os.path.join(self.wd.path, Pinocchio.PIN_REDSHIFT_FILE)

        with open(os.path.join(fp), "w") as temp_file:
            temp_file.write(self.redShiftRequest.header)
            k: str
            for k in self.redShiftRequest.redShifts:
                temp_file.write(k + "\n")

        return fp

    def __writeConfigToWD(self) -> str:
        """
        create a pinocchio config file and write it into the cwd    

        :return: path of the config on the filesystem
        :rtype: str
        """
        assert self.wd is not None

        lines: List[str] = []
        # add header
        lines.append(f"{Pinocchio.PRMS_CMNT} Generated param file for Pinocchio by Karabo Framework (https://github.com/i4Ds/Karabo-Pipeline)")
        lines.append("")

        # write entries
        k: str
        v: List[PinocchioParams]
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

    def save(self, outDirPath: str) -> None:
        """
        save pinocchio results after a run() into the given folder
        Overwrites all files with the same name within that folder
        Directory will not be created, needs to exist before this call

        :param outDirPath: path to the output folder
        :type outDirPath: str
        """
        
        assert self.didRun, "can not save pinocchio results if run() was never called"
        if not os.path.isdir(outDirPath):
            os.mkdir(outDirPath)
        
        shutil.copytree(self.wd.path, outDirPath, dirs_exist_ok=True)

    def plotHalos(self, redshift: str = "0.0", save: bool = False) -> None:
        """
        plotting the generated halos after a pinocchio run

        :param redshift: redshift that chooses the mass function file, defaults to "0.0"
        :type redshift: str, optional
        :param save: save the plot to the cwd, defaults to False
        :type save: bool, optional
        """
        assert self.didRun, ("can not plot mass function if run() was never called, "
                                "use the Pinocchio.plotMassFunctionFromFile() call instead")
        Pinocchio.plotHalosFromFile(self.outCatalogPath[redshift], save)

    @staticmethod
    def plotHalosFromFile(path: str, save: bool = False) -> None:
        """
        plot the halos given by a file - visualisation from pinocchio

        :param path: path to the catalog file
        :type path: str
        :param save: save the plot to the cwd, defaults to False
        :type save: bool, optional
        """

        (x,y,z) = np.loadtxt(path, unpack=True, usecols=(5,6,7))

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
        plotting the mass function after a pinocchio run

        :param redshift: redshift that chooses the mass function file, defaults to "0.0"
        :type redshift: str, optional
        :param save: save the plot to the cwd, defaults to False
        :type save: bool, optional
        """
        assert self.didRun, ("can not plot mass function if run() was never called, "
                                "use the Pinocchio.plotMassFunctionFromFile() call instead")
        Pinocchio.plotMassFunctionFromFile(self.outMFPath[redshift], save)

    @staticmethod
    def plotMassFunctionFromFile(path:str, save: bool = False) -> None:
        """
        plot the mass function given by a file - visualisation from pinocchio

        :param path: path of the massfunction file created by pinocchio
        :type path: str
        :param save: save the plot to the cwd, defaults to False
        :type save: bool, optional
        """

        (m, nm, fit) = np.loadtxt(path, unpack=True, usecols=(0,1,5))

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

    def plotPastLightCone(self, save: bool = False) -> None:
        """
        plotting the past light cone after a pinocchio run

        :param save: save the plot to the cwd, defaults to False
        :type save: bool, optional
        """
        assert self.didRun, ("can not plot past light cone if run() was never called, "
                                "use the Pinocchio.plotPastLightConeFromFile() call instead")
        Pinocchio.plotPastLightConeFromFile(self.outLightConePath, save)
    
    @staticmethod
    def plotPastLightConeFromFile(path: str, save: bool = False) -> None:
        """
        plot the past light cone given by a file - visualisation from pinocchio

        :param path: path to the plc file
        :type path: str
        :param save: save the plot to the cwd, defaults to False
        :type save: bool, optional
        """

        (x,y,z,m)=np.loadtxt(path, unpack=True, usecols=(2,3,4,8))

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

    def getSkyModel(self, near: float = 0, far: float = 100) -> SkyModel:
        """
        This method should be used on the pinocchio object after run() was called. 
        It is a wrapper for the getSkyModelFromFiles() static method with the correct 
        internal path. 

        :param near: near distance, everything below gets clipped, defaults to 0
        :type near: float, optional
        :param far: far distance, everything above gets clipped, defaults to 100
        :type far: float, optional
        :return: SkyModel with the point sources in radial distance from near to far
        :rtype: SkyModel
        """
        assert self.didRun, "can not get sky model if run() was never called"
        return Pinocchio.getSkyModelFromFiles(self.outLightConePath, near, far)

    @staticmethod
    def getSkyModelFromFiles(path: str, near: float = 0, far: float = 100) -> SkyModel:
        """
        Create a sky model from the pinocchio simulation cone. All halos from the near to 
        far plane (euclid distance) will be translated into the RA (right ascension - [0,360] in deg) and 
        DEC (declination - [0, 90] in deg) format. 
        
        example in 1D - this function will do the same on pinocchios 3D cone:

        |               10              60          |
        (0,0,0) --------near------------far--------->

        and translated into RA DEC

        :param path: path to the past light cone file, if there is no file, use the a pinocchio run 
                     to create such a file. 
        :type path: str
        :param near: starting distance from the (0,0,0) point in [Mpc/h], defaults to 0
        :type near: float, optional
        :param far: ending distance from the (0,0,0) point in [Mpc/h], default to 100
        :type far: float, optional
        :return: SkyModel with the point sources in radial distance from near to far
        :rtype: SkyModel
        """
    
        assert near < far, "near is further or equal to far"

        # load pinocchio data
        (x,y,z)=np.loadtxt(path, unpack=True, usecols=(2,3,4))    
        
        assert x.shape == y.shape == z.shape, "x, y, z do not have the same dimensions"
    
        length: int = len(x)
        skyModArr = np.empty((length, 7))

        i: int
        for i in range(length):
            xPin = x[i]
            yPin = y[i]
            zPin = z[i]

            radDist = np.sqrt(xPin**2 + yPin**2 + zPin**2) # radial distance

            # skip if not between near and far "plane"
            if radDist < near and radDist > far:
                continue

            # calculate RA DEC
            theta = np.arccos(zPin/radDist) # theta
            ra = np.arctan2(yPin, xPin) * Pinocchio.RAD_TO_DEG # RA
            dec = (np.pi/2. - theta) * Pinocchio.RAD_TO_DEG # DEC

            skyModArr[i, Pinocchio.PIN_RIGHT_ASCENSION_IDX] = ra
            skyModArr[i, Pinocchio.PIN_DECLINATION_IDX] = dec
            skyModArr[i, Pinocchio.PIN_I_FLUS_IDX] = 1
            skyModArr[i, Pinocchio.PIN_Q_FLUS_IDX] = 0
            skyModArr[i, Pinocchio.PIN_U_FLUS_IDX] = 0
            skyModArr[i, Pinocchio.PIN_V_FLUS_IDX] = 0
            skyModArr[i, Pinocchio.PIN_REF_F_IDX] = 1.e8

        return SkyModel(skyModArr)
