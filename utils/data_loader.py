import nest_asyncio
import re
import os
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cupy as cp


class DataLoader:
    """Load input data"""

    def __init__(self, path):
        self.path = path
        self.MAX_LEN = 128
        nest_asyncio.apply()

    @staticmethod
    def __rm_tags(text):
        re_tags = re.compile(r'<[^>]+>')
        return re_tags.sub(' ', text)

    def __read_files(self):
        file_list = []

        positive_path = self.path + "/pos/"
        for f in os.listdir(positive_path):
            file_list += [positive_path + f]

        negative_path = self.path + "/neg/"
        for f in os.listdir(negative_path):
            file_list += [negative_path + f]

        all_labels = ([1] * 12500 + [0] * 12500)
        all_texts = []
        for fi in file_list:
            with open(fi, encoding='utf8') as file_icput:
                all_texts += [self.__rm_tags(" ".join(file_icput.readlines()))]

        return all_labels, all_texts

    def load_data(self):

        y_train, train_text = self.__read_files()

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        sentences = train_text
        labels = y_train

        input_ids = [tokenizer.encode(sent, add_special_tokens=True, max_length=self.MAX_LEN, truncation=True)
                     for sent in sentences]
        input_ids = pad_sequences(input_ids, maxlen=self.MAX_LEN, dtype="long",
                                  value=0, truncating="post", padding="post")

        attention_masks = []

        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)

        train_inputs = input_ids
        train_labels = labels

        train_masks = attention_masks
        train_inputs = cp.array(train_inputs)
        train_labels = cp.array(train_labels)
        train_masks = cp.array(train_masks)

        return train_inputs, train_labels, train_masks
