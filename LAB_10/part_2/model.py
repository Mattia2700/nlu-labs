import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel


class ModelIAS(nn.Module):
    def __init__(self, intent_len, slot_len):
        super(ModelIAS, self).__init__()
        self.intent_len = intent_len
        self.slot_len = slot_len
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.intent_classifier = nn.Linear(
            self.bert.config.hidden_size, self.intent_len
        )
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, self.slot_len)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
    ):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        slot_logits = self.slot_classifier(outputs[0])
        intent_logits = self.intent_classifier(outputs[1])

        return slot_logits, intent_logits
