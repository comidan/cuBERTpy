from modules.linear_layer import LinearLayer
from modules.layer_norm import LayerNorm
from modules.dropout import Dropout


class BertSelfOutput:
    def __init__(self, config):
        self.dense = LinearLayer(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = Dropout(config.hidden_dropout_prob)

    def init_param(self, weights, biases):
        self.dense.init_param(weights[0], biases[0])
        self.LayerNorm.init_param(weights[1], biases[1])

    def __call__(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
