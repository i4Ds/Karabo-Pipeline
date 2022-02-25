from karabo.Container.environment import Environment


class KaraboException(Exception):
    """
    Base Exception thrown by the Karabo Pipeline
    """


class KaraboEnvironmentBuildException(KaraboException):
    """
    Exception for Errors occuring during the build process for a karabo environment to a container
    """

    def __init__(self, build_error_log: str, env: Environment):
        if "install" in build_error_log:
            self.message = f"Could not install specified packages: {' '.join(env.conda_packages)} with given channels" \
                           f" {' '.join(env.conda_channels)}. Are you missing channels? Check whether the packages exists on repo.anaconda.org"
        if "FROM" in build_error_log:
            self.message = f"base image could not be found. Is it correct?"

        if "ENTRYPOINT" in build_error_log:
            self.message = f"Entrypoint could not be set."

        if "create" in build_error_log:
            self.message = f"Could not create environment in container. Is the python version correct?"

            super(KaraboEnvironmentBuildException, self).__init__(self.message)
