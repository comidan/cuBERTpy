import cupy as cp


class Embedding:
    """The embedding layer module made from scratch."""

    def __init__(self, dim_in, dim_out, padding_idx=0):
        self.weight = cp.zeros(shape=(dim_in, dim_out))

    def init_param(self, weights):
        self.__set_weights(weights)

    def __set_weights(self, weights):
        self.weight = cp.array(weights)

    def __call__(self, x, reshape=False, axis=0):
        if reshape:
            return cp.expand_dims(cp.take(self.weight, cp.reshape(x, (-1,)), axis=axis), 0)
        else:
            return cp.take(self.weight, x, axis=axis)
