from attention.attention import BertAttention
from modules.intermediate import BertIntermediate
from modules.output import BertOutput


class BertLayer:
    def __init__(self, config):
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def init_param(self, weights, biases):
        self.attention.init_param(weights[0], biases[0])
        self.attention.output.init_param([weights[0][3], weights[2][0]], [biases[0][3], biases[2][0]])
        self.intermediate.init_param(weights[1][0], biases[1][0])
        self.output.init_param([weights[1][1], weights[2][1]], [biases[1][1], biases[2][1]])

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs

        return outputs
