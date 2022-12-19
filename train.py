import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from model import EntityClsModel
from data_helper import EntityClsDataset

lr = 2e-5
batch_size = 2
epochs = 10
max_len=20
hidden_size=768
train_path = "./data/data_train.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
save_model_path = "./save_model/best_weights.pth"
label2idx_path = "./data/label2idx.json"

with open(label2idx_path, encoding="utf-8") as f:
    label2idx = json.loads(f.read())
cls_num = len(label2idx)

def train():
    dataset = EntityClsDataset(data_path=train_path, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    model = EntityClsModel(cls_num=cls_num, hidden_size=hidden_size)

    optimizer = AdamW(params=model.parameters(), lr=lr)

    model.train()
    model.to(device)

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch = [d.to(device) for d in batch]
            output = model(*batch[:-1])
            loss = criterion(output, batch[-1])
            loss.backward()
            optimizer.step()
            print(loss.item())

    torch.save(model.state_dict(), save_model_path)

if __name__ == '__main__':
    train()


