# Add functions or classes used for data loading and preprocessing
import os
import json
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from collections import Counter
from transformers import BertTokenizer


def load_data(path):
    """
    input: path/to/data
    output: json
    """
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


class IntentsAndSlots(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, pad_length=50, unk="unk"):
        self.utterances = []
        self.intent_label = []
        self.slot_labels = []
        self.unk = unk
        self.intents = []
        self.slots = []
        self.pad_length = pad_length

        for x in dataset:
            if x["intent"] not in self.intents:
                self.intents.append(x["intent"])
            for s in x["slots"]:
                if s not in self.slots:
                    self.slots.append(s)

        for x in dataset:
            self.utterances.append(x["utterance"].split())
            self.intent_label.append(self.intents.index(x["intent"]))
            self.slot_labels.append([self.slots.index(s) for s in x["slots"]])

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = self.utterances[idx]
        slots = self.slot_labels[idx]
        intent = self.intent_label[idx]

        tokens = []
        slot_labels_ids = []
        for u, s in zip(utt, slots):
            u_token = self.tokenizer.tokenize(u)
            if not u_token:
                print(self.tokenizer.unk_token)
                tokens = [self.tokenizer.unk_token]
            tokens.extend(u_token)
            slot_labels_ids.extend([s] + [0] * (len(u_token) - 1))

        assert len(tokens) < self.pad_length - 2

        tokens += [self.tokenizer.sep_token]
        slot_labels_ids += [0]
        token_type_ids = [0] * len(tokens)

        tokens = [self.tokenizer.cls_token] + tokens
        slot_labels_ids = [0] + slot_labels_ids
        token_type_ids = [0] + token_type_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        padding_length = self.pad_length - len(input_ids)
        input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        slot_labels_ids = slot_labels_ids + ([0] * padding_length)

        sample = {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "token_type_ids": torch.LongTensor(token_type_ids),
            "intent_label_ids": torch.as_tensor(int(intent)),
            "slot_labels_ids": torch.LongTensor(slot_labels_ids),
        }

        return sample

    # Auxiliary methods

    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data, mapper):  # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res


def load_dataset():
    tmp_train_raw = load_data(os.path.join("dataset", "ATIS", "train.json"))
    test_raw = load_data(os.path.join("dataset", "ATIS", "test.json"))
    return tmp_train_raw, test_raw


def split_dataset(tmp_train_raw, test_raw):
    # Firt we get the 10% of dataset, then we compute the percentage of these examples
    # on the training set which is around 11%
    portion = round(
        ((len(tmp_train_raw) + len(test_raw)) * 0.10) / (len(tmp_train_raw)), 2
    )

    intents = [x["intent"] for x in tmp_train_raw]  # We stratify on intents
    count_y = Counter(intents)

    Y = []
    X = []
    mini_Train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1:  # If some intents occure once only, we put them in training
            X.append(tmp_train_raw[id_y])
            Y.append(y)
        else:
            mini_Train.append(tmp_train_raw[id_y])

    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(
        X, Y, test_size=portion, random_state=42, shuffle=True, stratify=Y
    )
    X_train.extend(mini_Train)
    train_raw = X_train
    val_raw = X_dev

    y_test = [x["intent"] for x in test_raw]

    return train_raw, val_raw, test_raw
