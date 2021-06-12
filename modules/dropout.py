import cupy as cp


class Dropout:
    """Dropout layer implementation using binomial distribution"""

    def __init__(self, p, evaluation=True):
        self.p = p
        self.evaluation = evaluation

    def __call__(self, x):
        if not self.evaluation:
            binary_value = cp.random.uniform(size=(x.shape[-2], x.shape[-1])) > self.p
            res = cp.multiply(x, binary_value)
        else:
            res = x
        return res
