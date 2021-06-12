import cupy as cp


class LinearLayer:
    """Linear Layer implementation from scratch"""

    def __init__(self, in_features, out_features, bias=None):
        self.weights = cp.ndarray(shape=(in_features, out_features))
        self.bias = bias

    def init_param(self, weights, bias):
        self.__set_weights(weights)
        self.__set_bias(bias)

    def __set_weights(self, weights):
        self.weights = cp.array(weights)

    def __set_bias(self, bias):
        self.bias = cp.array(bias)

    def __call__(self, x):
        if self.bias is not None:
            return cp.dot(x, cp.transpose(self.weights)) + self.bias
        else:
            return cp.dot(x, cp.transpose(self.weights))
