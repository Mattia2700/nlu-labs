# Add functions or classes used for data loading and preprocessing
import nltk

# nltk.download("subjectivity")
# from nltk.corpus import subjectivity
import random
from torch.utils.data import Dataset
from transformers import BertTokenizer


class Sentences(Dataset):
    def __init__(self, sentences, ref):
        self.sentences = sentences
        self.ref = ref
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.ref[idx]
