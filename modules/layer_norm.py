import cupy as cp


class LayerNorm:
    """Layer Normalization module implementation from scratch"""

    def __init__(self, features_dim=768, epsilon=1e-12):
        self.gamma = cp.ones(features_dim)
        self.beta = cp.zeros(features_dim)
        self.epsilon = epsilon

    def init_param(self, weight, bias):
        self.__set_gamma(weight)
        self.__set_beta(bias)

    def __set_gamma(self, gamma):
        self.gamma = cp.array(gamma)

    def __set_beta(self, beta):
        self.beta = cp.array(beta)

    def __call__(self, x):
        mean = cp.mean(x, axis=-1, keepdims=True)
        var = cp.mean(((x - mean) ** 2), axis=-1, keepdims=True)
        std = cp.sqrt((var + self.epsilon))
        y = (x - mean) / std
        if self.gamma is not None:
            y *= self.gamma
        if self.beta is not None:
            y += self.beta
        return y
