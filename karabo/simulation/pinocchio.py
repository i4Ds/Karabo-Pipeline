import os
import subprocess
import shutil
from dataclasses import dataclass, field
from typing import List

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

    PIN_DEFAULT_PARAMS_FILE = "pinocchio_params.conf"
    PIN_DEFAULT_OUTPUT_FILE = "pinocchio_outs.conf"
    PIN_PARAM_FILE          = "parameter_file"
    PIN_REDSHIFT_FILE       = "outputs"

    PRMS_CMNT = "#"
    PRMS_CMNT_SPLITTER = "%"

    PIN_OUT_VAL_PADD = 3                            # padding after the name
    PIN_OUT_COMM_PADD = 12 + PIN_OUT_VAL_PADD       # padding between name and comment

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

    def __loadPinocchioDefaultRedShiftRequest__(self) -> PinocchioRedShiftRequest:
        return self.loadPinocchioRedShiftRequest(self.redShiftInputPath) 

    def loadPinocchioRedShiftRequest(self, path: str) -> PinocchioRedShiftRequest:
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

    def addRedShift(self, rs: str):
        """
        Add a redshift to the outputs file, 0.0 is default
        """

        assert len(rs.split(".")) == 2, "Input is no double value"
        self.redShiftRequest.redShifts.append(rs)

    def removeRedShift(self, rs: str):
        """
        Remove a redshift from the outputs file
        """

        assert len(rs.split(".")) == 2, "Input is no double value"
        self.redShiftRequest.redShifts.remove(rs)

    def __loadPinocchioDefaultConfig__(self) -> PinocchioConfig:
        return self.loadPinocchioConfig(self.paramsInputPath)     

    def loadPinocchioConfig(self, path: str) -> PinocchioConfig:
        """
        Load standard Pinocchio config from the installed package
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

    def run(self, printLiveOutput = False) -> None:
        """
        run pinocchio in a temp folder
        """

        assert len(self.redShiftRequest.redShifts) > 0, "all redshifts removed from outputs file - pinocchio won't calculate anything"

        self.runConfigPath = self.__writeConfigToWD__()
        self.outputFilePath = self.__writeRedShiftRequestFileToWD__()

        cmd: List[str] = ["pinocchio", self.runConfigPath]
        self.out = subprocess.run(cmd, cwd=self.wd.path, capture_output = not printLiveOutput, text=True) 

    def printPinocchioStdOutput(self):
        if hasattr(self, "out"):
            print(self.out.stdout)

    def printPinocchioStdError(self):
         if hasattr(self, "out"):
            print(self.out.stderr)

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

    def save(self, outDir: str):
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

    def getRunplannerOutput(self):
        pass

