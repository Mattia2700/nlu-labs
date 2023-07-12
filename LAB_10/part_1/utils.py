# Add functions or classes used for data loading and preprocessing
import os
import json
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from collections import Counter


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
    def __init__(self, dataset, lang, unk="unk"):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk

        for x in dataset:
            self.utterances.append(x["utterance"])
            self.slots.append(x["slots"])
            self.intents.append(x["intent"])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {"utterance": utt, "slots": slots, "intent": intent}
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
    tmp_train_raw = load_data(
        os.path.join("dataset", "IntentSlotDatasets", "ATIS", "train.json")
    )
    test_raw = load_data(
        os.path.join("dataset", "IntentSlotDatasets", "ATIS", "test.json")
    )
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
