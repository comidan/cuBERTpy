import torch


class LoadModel:
    """Class for loading model from a saved state of a PyTorch one"""

    def __init__(self, path):
        self.path = path
        self.layer_id = "bert.encoder.layer."
        self.embedding_params = ["bert.embeddings.position_ids",
                                 "bert.embeddings.word_embeddings.weight",
                                 "bert.embeddings.position_embeddings.weight",
                                 "bert.embeddings.token_type_embeddings.weight",
                                 "bert.embeddings.LayerNorm.weight",
                                 "bert.embeddings.LayerNorm.bias"]
        self.feed_forward_nn_params = ["intermediate.dense.weight", "intermediate.dense.bias",
                                       "output.dense.weight", "output.dense.bias"]
        self.layer_norm_params = ["attention.output.LayerNorm.weight", "attention.output.LayerNorm.bias",
                                  "output.LayerNorm.weight", "output.LayerNorm.bias"]
        self.encoder_params = ["attention.self.query.weight", "attention.self.query.bias",
                               "attention.self.key.weight", "attention.self.key.bias",
                               "attention.self.value.weight", "attention.self.value.bias",
                               "attention.output.dense.weight", "attention.output.dense.bias"]
        self.output_params = ["bert.pooler.dense.weight", "bert.pooler.dense.bias",
                              "classifier.weight", "classifier.bias"]
        self.model = torch.load(self.path, map_location='cuda:0')

    @staticmethod
    def __load_tensor(tensor, log):

        def log_quantization(x):
            x = x.to(torch.float32)
            x = torch.where(x > 0, torch.pow(2, torch.round(torch.log2(torch.abs(x)))), x)
            x = torch.where(x < 0, -torch.pow(2, torch.round(torch.log2(torch.abs(x)))), x)
            return x

        if log:
            return log_quantization(tensor.cpu().detach()).numpy()
        else:
            return tensor.cpu().detach().numpy()

    def __load_params(self, layer, params, log):
        """Return requested parameters"""
        out = [self.__load_tensor(self.model.get(module), log) if layer is None else
               self.__load_tensor(self.model.get(self.layer_id + str(layer) + "." + module), log)
               for module in params]
        return out[::2], out[1::2]

    def __load_embeddings_params(self):
        """Return embeddings parameter"""
        out = [self.__load_tensor(self.model.get(param), False) for param in self.embedding_params]
        return out[0:len(out) - 1], out[-1]

    def get_model_depth(self):
        """Return model depth"""
        layers = [int(name.replace(self.layer_id, "")[0])
                  for name, param in self.model.items()
                  if self.layer_id in name]
        return max(layers) + 1

    def load_param(self, log=False):
        """Return structured weights and biases tensors in numpy form"""

        n_layers = self.get_model_depth()

        weights = []
        biases = []

        w_embeddings, norm_b = self.__load_embeddings_params()
        weights.append([w_embeddings])
        biases.append([norm_b])

        for i in range(n_layers):
            w_self, b_self = self.__load_params(i, self.encoder_params, log)
            w_ff, b_ff = self.__load_params(i, self.feed_forward_nn_params, log)
            w_norm, b_norm = self.__load_params(i, self.layer_norm_params, log)
            weights.append([w_self, w_ff, w_norm])
            biases.append([b_self, b_ff, b_norm])

        w_out, b_out = self.__load_params(None, self.output_params, log)
        weights.append(w_out)
        biases.append(b_out)

        return weights, biases
