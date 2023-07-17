import torch
import torch.nn as nn
from transformers import BertModel


class Subjectivity(nn.Module):
    def __init__(self):
        super(Subjectivity, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        fc_output = self.fc(output[1])
        return self.sigmoid(fc_output).squeeze()
