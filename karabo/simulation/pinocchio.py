import os
from dataclasses import dataclass

from matplotlib import lines
from karabo.util.FileHandle import FileHandle

@dataclass
class ParamInput:
    name: str
    isFlag: bool
    active: bool
    value: str
    comment: str

class Pinocchio:

    PIN_PARAMS_FILE = "pinocchio_params.conf"
    PIN_OUTPUT_FILE = "pinocchio_outs.conf"

    PRMS_CMNT = "#"
    PRMS_CMNT_SPLITTER = "%"

    def __init__(self):
        """
        Creates temp directory (wd) for the pinocchio run. 

        """
        self.wd = FileHandle(is_dir = True)
        
        # get default input files
        inputFilesPath = os.environ["CONDA_PREFIX"] + "/etc/"
        self.paramsInputPath = inputFilesPath + Pinocchio.PIN_PARAMS_FILE
        self.outputsInputPath = inputFilesPath + Pinocchio.PIN_OUTPUT_FILE

        self.currConfig: dict[str, list[ParamInput]] = self.__loadPinocchioConfigs__()

    def __loadPinocchioConfigs__(self) -> dict[str, list[ParamInput]]:
        configF = open(self.paramsInputPath)
        
        # remove header
        line: str = configF.readline()
        if line[0] != Pinocchio.PRMS_CMNT:
            print("input file is broken or has no header")
            return {}
        
        configMap: dict[str, list[ParamInput]] = {}
        currMapName: str = ""

        while line:
            line = configF.readline()
           
            # skip empty lines
            if len(line.strip()) == 0:
                continue
            
            # header found
            if line[0] == Pinocchio.PRMS_CMNT and line[1] == " ":
                currMapName = line[2:].strip()
                configMap[currMapName] = [] 
                continue

            commentSplit: list[str] = line.split(Pinocchio.PRMS_CMNT_SPLITTER)
            
            # empty comment line
            if len(commentSplit) == 2 and commentSplit[0].strip() == "":
                continue

            comment: str = commentSplit[-1].strip()
            paramPart: list[str] = commentSplit[:-1]
            lineSplit: list[str] = "".join(paramPart).split()

            # deactivated flag found
            if len(paramPart) == 2 and len(lineSplit) == 1:
                configMap[currMapName].append(ParamInput(lineSplit[0], True, False, "", comment))
            
            # deactivated param found
            elif len(paramPart) == 2 and len(lineSplit) == 4:
                configMap[currMapName].append(ParamInput(lineSplit[0], False, False, " ".join(lineSplit[1:]), comment))
                continue

            # activated flag found
            elif len(lineSplit) == 1:
                configMap[currMapName].append(ParamInput(lineSplit[0], True, True, "", comment))

            # param with value found
            elif len(lineSplit) == 2:
                configMap[currMapName].append(ParamInput(lineSplit[0], False, True, lineSplit[1], comment))

            else:
                assert False, "invalid entry"

        configF.close()
        return configMap

    def getConfig(self) -> dict[str, any]:
        """
        Get pinocchio run config as a dictionary
        
        The first call returns the default dict that can be changed and set with 
        the setConfig method
        """        
        return self.currConfig
    
    def setConfig(self, config : dict[str, str]) -> None:
        """
        Replace the default pinocchio config with the given config
        """
        pass

    def setOwnConfig(self, file: str) -> None:
        pass

    def getRunplannerOutput(self):
        pass



