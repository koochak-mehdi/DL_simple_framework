import numpy as np

class Constant:
    def __init__(self, con_value=.1) -> None:
        self.con_value = con_value
        if not (con_value < 1) and not (con_value >= 0):
            self.con_value = .1

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.con_value)


class UniformRandom:
    def __init__(self) -> None:
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.sample(weights_shape)

class Xavier:
    def __init__(self) -> None:
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        delta = np.sqrt(2/(fan_in + fan_out))
        return np.random.randn(*weights_shape) * delta

class He:
    def __init__(self) -> None:
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        delta = np.sqrt(2/fan_in)
        return np.random.randn(*weights_shape) * delta