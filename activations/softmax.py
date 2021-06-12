import cupy as cp


class Softmax:

    def __call__(self, x, dim=None):
        """Compute softmax values for each sets of scores in x according to axis"""

        e_x = cp.exp(x - cp.max(x, axis=dim, keepdims=True))
        return e_x / e_x.sum(axis=dim, keepdims=True)
