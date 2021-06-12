from modules.linear_layer import LinearLayer
from activations.gelu import GELU


class BertIntermediate:
    def __init__(self, config):
        self.dense = LinearLayer(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = GELU()

    def init_param(self, weights, biases):
        self.dense.init_param(weights, biases)

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
