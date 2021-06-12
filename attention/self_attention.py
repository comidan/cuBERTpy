from embeddings.embedding import Embedding
from modules.linear_layer import LinearLayer
from modules.dropout import Dropout
from activations.softmax import Softmax
import cupy as cp
import math


class BertSelfAttention:
    def __init__(self, config):
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = LinearLayer(config.hidden_size, self.all_head_size)
        self.key = LinearLayer(config.hidden_size, self.all_head_size)
        self.value = LinearLayer(config.hidden_size, self.all_head_size)

        self.dropout = Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def init_param(self, weights, biases):
        self.query.init_param(weights[0], biases[0])
        self.key.init_param(weights[1], biases[1])
        self.value.init_param(weights[2], biases[2])

    def transpose_for_scores(self, x):
        batch_size = x.shape[0]
        return cp.transpose(cp.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size)),
                            (0, 2, 1, 3))

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
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = cp.matmul(query_layer, cp.transpose(key_layer, (0, 1, 3, 2)))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.shape[1]
            position_ids_l = cp.reshape(cp.arange(seq_length, dtype=cp.int64), (-1, 1))
            position_ids_r = cp.reshape(cp.arange(seq_length, dtype=cp.int64), (1, -1))
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = cp.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = cp.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = cp.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = Softmax()(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = cp.matmul(attention_probs, value_layer)

        batch_size = query_layer.shape[0]
        context_layer = cp.reshape(cp.transpose(context_layer, (0, 2, 1, 3)),
                                   (batch_size, -1, self.num_attention_heads * self.attention_head_size))

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
