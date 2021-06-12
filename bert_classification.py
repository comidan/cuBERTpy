from bert import BertModel
from modules.dropout import Dropout
from modules.linear_layer import LinearLayer


class BertForSequenceClassification:
    def __init__(self, config):
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = LinearLayer(config.hidden_size, config.num_labels)

    def init_param(self, weight, bias):
        self.bert.init_param(weight[0:self.config.num_hidden_layers + 1],
                             bias[0:self.config.num_hidden_layers + 1])
        self.bert.pooler.init_param(weight[self.config.num_hidden_layers + 1][0],
                                    bias[self.config.num_hidden_layers + 1][0])
        self.classifier.init_param(weight[self.config.num_hidden_layers + 1][1],
                                   bias[self.config.num_hidden_layers + 1][1])

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        output = (logits,) + outputs[2:]
        return output
