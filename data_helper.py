import json
import os
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset


class EntityClsDataset(Dataset):
    def __init__(self, data_path, max_len, label2idx_path="./data/label2idx.json",
                 pretrained_model="bert-base-chinese"):
        super(EntityClsDataset, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.max_len = max_len
        self.datas = []
        labels = set()
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                text = line["text"]
                for entiry_info in line["entitys"]:
                    if entiry_info["start"]+1+len(entiry_info["entity"]) > max_len - 2:
                        continue
                    self.datas.append([text, entiry_info])
                    labels.add(entiry_info["label"])

        if os.path.exists(label2idx_path):
            with open(label2idx_path, encoding="utf-8") as f:
                self.label2idx = json.loads(f.read())
        else:
            labels = list(labels)
            self.label2idx = {label: idx for idx, label in enumerate(labels)}
            with open(label2idx_path, "w", encoding="utf-8") as f:
                json.dump(self.label2idx, f, ensure_ascii=False)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        text, entity_info = data
        inputs = self.tokenizer.encode_plus(text,
                                            return_tensors="pt",
                                            padding="max_length",
                                            truncation=True,
                                            max_length=self.max_len)
        label = self.label2idx[entity_info["label"]]
        entity_mask = torch.zeros(self.max_len)
        start = entity_info["start"]
        entity = entity_info["entity"]
        entity_mask[start+1: start+len(entity)+1] = 1
        input_ids = inputs["input_ids"][0]
        token_type_ids = inputs["token_type_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        return input_ids, attention_mask, token_type_ids, entity_mask, label


if __name__ == '__main__':
    dataset = EntityClsDataset("./data/data_train.json", max_len=20)
    dataset.__getitem__(0)
    dataset.__getitem__(2)






