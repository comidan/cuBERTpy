from activations.softmax import Softmax
from bert_classification import BertForSequenceClassification
from utils.load_model import LoadModel
from utils.config import BertConfig
import cupy as cp
from utils.data_loader import DataLoader
from tqdm import tqdm

if __name__ == "__main__":
    path = "./data/state1_reg.pt"
    load_model = LoadModel(path)
    print("Model loading...", end="")
    n_layers = load_model.get_model_depth()
    weights, biases = load_model.load_param(log=True)

    config = BertConfig(attention_probs_dropout_prob=0.1,
                        gradient_checkpointing=False,
                        hidden_act="gelu",
                        hidden_dropout_prob=0.1,
                        hidden_size=768,
                        initializer_range=0.02,
                        intermediate_size=3072,
                        layer_norm_eps=1e-12,
                        max_position_embeddings=512,
                        model_type="bert",
                        num_attention_heads=12,
                        num_hidden_layers=n_layers,
                        num_labels=2,
                        pad_token_id=0,
                        type_vocab_size=2,
                        vocab_size=30522)

    bert = BertForSequenceClassification(config)
    bert.init_param(weights, biases)
    print("complete.")

    loader = DataLoader("F:/aclImdb/test")

    print("Tokenizing data...", end="")
    text_input, labels, mask_input = loader.load_data()
    print("complete.")

    text_input = [cp.expand_dims(text_input[i], axis=0) for i in range(len(text_input))]
    mask_input = [cp.expand_dims(mask_input[i], axis=0) for i in range(len(mask_input))]

    print("Model evaluation starting...")

    softmax = Softmax()
    result = cp.empty(shape=(len(text_input)))
    for i in tqdm(range(len(text_input))):
        output = bert(text_input[i], mask_input[i])
        label_pred = softmax(output[0], 1).argmax(1)
        result[i] = float((label_pred == labels[i]))
    accuracy = result.mean()
    print("Accuracy: ", accuracy)
