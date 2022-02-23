import io
import docker
from docker.errors import BuildError
from karabo.Container.environment import Environment
from karabo.error import KaraboEnvironmentBuildException


class Container:

    def __init__(self, env: Environment):
        self.local_image_name: str = ""
        self.env = env
        self.createEnvironmentInDockerContainer()

    def createEnvironmentInDockerContainer(self):
        """
        Create a Dockerfile from a given in-memory Conda Environment
        :param env: Conda environment
        :param start_command: command to execute on container start
        :return: image hash
        """
        base_command = f"FROM continuumio/miniconda3 \n"
        create_env_command = f"RUN conda create -n {self.env.name} python={self.env.python_version} \n"
        shell_command = f"SHELL [\"conda\", \"run\", \"-n\", \"{self.env.name}\", \"/bin/bash\", \"-c\"] \n"
        install_command = f"RUN conda install -y {'-c' if len(self.env.conda_channels) > 0 else ''} " \
                          f"{' -c '.join(self.env.conda_channels)}  {' '.join(self.env.conda_packages)} \n"
        run_command = f"ENTRYPOINT {self.env.start_script}"

        full_command_sequence = base_command + create_env_command + shell_command + install_command + run_command
        # create temporary dockerfile
        dockerfile = io.BytesIO(full_command_sequence.encode('utf-8'))

        client = docker.DockerClient.from_env()
        try:
            (image, logs) = client.images.build(path=".", fileobj=dockerfile, tag=f"karabo_container_env_{self.env.name}")
            self.local_image_name = image
            print(f"Container environment successfully created with image name: {image}")
        except BuildError as bld_error:
            raise KaraboEnvironmentBuildException(bld_error.msg, self.env)
        client.close()

    def run(self):
        if self.local_image_name != "":
            client = docker.DockerClient.from_env()
            log = client.containers.run(self.local_image_name)
            print(log)
            client.close()
