from embeddings.embedding import Embedding
from modules.layer_norm import LayerNorm
from modules.dropout import Dropout
import cupy as cp


class BertEmbeddings:
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = cp.empty(shape=config.max_position_embeddings)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def init_param(self, weight, bias):
        self.position_ids = cp.array(weight[0][0])
        self.word_embeddings.init_param(weight[0][1])
        self.position_embeddings.init_param(weight[0][2])
        self.token_type_embeddings.init_param(weight[0][3])
        self.LayerNorm.init_param(weight[0][4], bias[0])

    def __call__(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = cp.zeros(input_shape, dtype=cp.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)

        embeddings = self.dropout(embeddings)
        return embeddings
