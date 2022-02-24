class EastNorthCoordinate:

    def __init__(self, x: float, y: float, z: float = 0, x_error: float = 0, y_error: float = 0, z_error: float = 0):
        """
        class defining a coordinate set in north-east-(up) coordinates
        :param x: east coordinate
        :param y: north coordinate
        :param z: up coordinate
        :param x_error: east coordinate error
        :param y_error: north coordinate error
        :param z_error: up coordinate error
        """
        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.x_error: float = x_error
        self.y_error: float = y_error
        self.z_error: float = z_error
