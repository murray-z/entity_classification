import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertModel


class EntityClsModel(nn.Module):
    def __init__(self, hidden_size=768, cls_num=2, pretrained_model="bert-base-chinese", drop=0.1):
        super(EntityClsModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(hidden_size*2, cls_num)

    def forward(self, input_ids, attention_mask, token_type_ids, entity_mask):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        last_hidden_state, pooler_output = outputs.last_hidden_state, outputs.pooler_output
        entity_len = torch.sum(entity_mask, dim=1, keepdim=True)
        entity_mask = entity_mask.unsqueeze(-1)
        entity_hidden = last_hidden_state * entity_mask
        entity_hidden_sum = torch.sum(entity_hidden, dim=1)
        entity_hidden_mean = entity_hidden_sum / entity_len
        hidden = torch.cat([entity_hidden_mean, pooler_output], dim=1)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return output



