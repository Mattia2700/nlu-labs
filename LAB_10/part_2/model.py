import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# BERT model script from: huggingface.co
from transformers import BertModel

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

        if bidirectional:
            print(
                "When setting bidirectionality, the slot size is double the hidden size"
            )
            self.slot_out = nn.Linear(hid_size * 2, out_slot)
        else:
            self.slot_out = nn.Linear(hid_size, out_slot)

        self.intent_out = nn.Linear(hid_size, out_int)

        # Dropout layer How do we apply it?
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(
            utterance
        )  # utt_emb.size() = batch_size X seq_len X emb_size
        utt_emb = utt_emb.permute(
            1, 0, 2
        )  # we need seq len first -> seq_len X batch_size X emb_size

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost

        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy())
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)
        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output)
        # Get the last hidden state
        last_hidden = last_hidden[-1, :, :]

        if self.dropout:
            utt_encoded = self.dropout(utt_encoded)
            last_hidden = self.dropout(last_hidden)

        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)

        # Slot size: seq_len, batch size, calsses
        slots = slots.permute(1, 2, 0)  # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len

        return slots, intent
    
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertModel.from_pretrained("bert-base-uncased")

# inputs = tokenizer("I saw a man with a telescope", return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state

class CustomBERT(nn.Module):
    def __init__(self, out_slot, out_int):
        super(CustomBERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.slot_out = nn.Linear(self.bert.config.hidden_size, out_slot)
        self.intent_out = nn.Linear(self.bert.config.hidden_size, out_int)

    def forward(self, utterance, seq_lengths):
        outputs = self.bert(utterance)
        last_hidden_states = outputs.last_hidden_state
        slots = self.slot_out(last_hidden_states)
        intent = self.intent_out(last_hidden_states[:, 0, :])
        slots = slots.permute(1, 2, 0)
        print(intent)
        return slots, intent