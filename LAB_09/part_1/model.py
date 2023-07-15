import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable


class LM(nn.Module):
    @staticmethod
    def cosine_similarity(x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    def __init__(
        self,
        emb_size,
        hidden_size,
        output_size,
        pad_index=0,
        out_dropout=0.3,
        emb_dropout=0.3,
        n_layers=1,
        lstm=False,
        dropout=False,
    ):
        self.dropout = dropout
        super(LM, self).__init__()

        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        if self.dropout:
            self.embedding_dropout = nn.Dropout(emb_dropout)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        if lstm:
            self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)
        else:
            self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False)
        self.pad_token = pad_index
        if self.dropout:
            self.rnn_dropout = nn.Dropout(out_dropout)
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        if self.dropout:
            emb = self.embedding_dropout(emb)
        rnn_out, _ = self.rnn(emb)
        if self.dropout:
            rnn_out = self.rnn_dropout(rnn_out)
        output = self.output(rnn_out).permute(0, 2, 1)
        return output

    def get_most_similar(self, vector, top_k=10):
        embs = self.embedding.weight.detach().cpu().numpy()
        # Our function that we used before
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(LM.cosine_similarity(x, vector))
        # Take ids of the most similar tokens
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[indexes]
        return (indexes, top_scores)


class Lang:
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output
