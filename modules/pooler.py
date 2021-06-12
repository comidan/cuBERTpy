from modules.linear_layer import LinearLayer
import cupy as cp


class BertPooler:
    def __init__(self, config):
        self.dense = LinearLayer(config.hidden_size, config.hidden_size)
        self.activation = cp.tanh

    def init_param(self, weights, biases):
        self.dense.init_param(weights, biases)

    def __call__(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
