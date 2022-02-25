from enum import Enum

class ContainerBackend(Enum):
    uDocker = 0
    Docker = 1
    Singularity = 2


class Environment:
    def __init__(self, name: str, python_version: float, start_script: str):
        self.name: str = name
        self.conda_packages: [str] = []
        self.conda_channels: [str] = []
        self.start_script: str = start_script
        self.python_version: float = python_version

    def addPackage(self, package_name):
        if package_name not in self.conda_packages:
            self.conda_packages.append(package_name)

    def addChannel(self, channel_name):
        if channel_name not in self.conda_channels:
            self.conda_channels.append(channel_name)

    # def createEnvironment(self, backend: ContainerBackend):
    #     if backend == 1:
    #         cont = container.Container("continuumio/miniconda3")
    #         cont.createEnvironmentInDockerContainer(self)
    #     else:
    #         raise NotImplemented("Other Backends are not implemented yet")
    #         pass

