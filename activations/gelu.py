from cupyx.scipy.special._erf import erf


class GELU:
    """GELU, activation function used by BERT instead of RELU"""

    def __call__(self, x):
        sqrt_two = 1.4142135623730951
        return x * 0.5 * (1.0 + erf(x / sqrt_two))
