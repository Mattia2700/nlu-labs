import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertTokenizer

class ModelIAS(nn.Module):
    def __init__(
        self,
        hid_size,
        out_slot,
        out_int,
        emb_size,
        vocab_len,
        n_layer=1,
        pad_index=0,
        bidirectional=False,
        dropout=None,
    ):
        super(ModelIAS, self).__init__()

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        self.utt_encoder = nn.LSTM(
            emb_size, hid_size, n_layer, bidirectional=bidirectional
        )

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.slot_out = nn.Linear(self.bert.config.hidden_size, out_slot)

        self.intent_out = nn.Linear(self.bert.config.hidden_size, out_int)

        # Dropout layer How do we apply it?
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, utterance, utterance_mask):

        bert_out = self.bert(utterance, attention_mask=utterance_mask)
        sequence_output = bert_out[0]
        pooled_output = bert_out[1]

        if self.dropout:
            sequence_output = self.dropout(sequence_output)
            pooled_output = self.dropout(pooled_output)

        # Compute slot logits
        slots = self.slot_out(sequence_output)
        # Compute intent logits
        intent = self.intent_out(pooled_output)

        # Slot size: seq_len, batch size, calsses
        slots = slots.permute(0, 2, 1)  # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        
        # remove first and last token from slots
        slots = slots[:, :, 1:-1]

        return slots, intent
