import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class BERT(nn.Module):
    def __init__(self, config, device, max_length, out_class):
        super(BERT, self).__init__()

        self.max_length = max_length
        self.device = device
        self.dim = 768

        self.tokenizer = BertTokenizer.from_pretrained(config)
        self.bert = BertModel.from_pretrained(config).to(self.device)
        self.ffnn = nn.Sequential(
            nn.Linear(self.dim, self.dim // 16),
            nn.Dropout(0.2),
            nn.Linear(self.dim // 16, 32),
            nn.Dropout(0.2),
            nn.Linear(32, out_class)
        )

    def forward(self, sentence):
        input = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = torch.tensor(input['input_ids'], dtype=torch.long).to(self.device)
        input_att = torch.tensor(input['attention_mask'], dtype=torch.long).to(self.device)
        vector = self.bert(input_ids, attention_mask=input_att)[0].permute(1, 0, 2)[0]
        output = self.ffnn(vector)
        return output
