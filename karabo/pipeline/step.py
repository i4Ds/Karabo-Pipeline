from karabo.Container.environment import Environment


class PipelineStep:
    def __init__(self):
        self.inputs: [(str, object)]
        self.outputs: [(str, object)]
        self.Environment: Environment
        self.command: str
