import json

import torch
from model import EntityClsModel
from transformers import BertTokenizer

save_model_path = "./save_model/best_weights.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
pretrained_model="bert-base-chinese"
label2idx_path = "./data/label2idx.json"

tokenizer = BertTokenizer.from_pretrained(pretrained_model)
with open(label2idx_path, encoding="utf-8") as f:
    label2idx = json.loads(f.read())
idx2label = {idx: label for label, idx in label2idx.items()}

model = EntityClsModel()
model.load_state_dict(torch.load(save_model_path))

model.to(device)
model.eval()

def encode(query):
    text = query["text"]
    inputs = tokenizer.encode_plus(text, return_tensors="pt")
    num_entity = len(query["entitys"])
    input_ids = inputs["input_ids"].repeat(num_entity, 1)
    token_type_ids = inputs["token_type_ids"].repeat(num_entity, 1)
    attention_mask = inputs["attention_mask"].repeat(num_entity, 1)

    input_shape = input_ids.size()

    entity_mask = torch.zeros(input_shape[0], input_shape[1])

    for idx, entity_info in enumerate(query["entitys"]):
        start = entity_info["start"]
        entity = entity_info["entity"]
        entity_mask[idx][start + 1: start + len(entity) + 1] = 1

    return input_ids, attention_mask, token_type_ids, entity_mask



def predict(query):
    encoded = encode(query)
    names = [info["entity"] for info in query["entitys"]]
    encoded = [d.to(device) for d in encoded]
    output = model(*encoded)
    output = list(torch.argmax(output, dim=1).cpu().numpy())
    res = {}
    for n, o in zip(names, output):
        label = idx2label[o]
        res[n] = label
    print(res)



if __name__ == '__main__':
    test = {"text": "查询下哈尔滨机场到杭州机场的航班",
            "entitys": [{"entity": "哈尔滨机场", "start": 3},
                        {"entity": "杭州机场", "start": 9}]}

    print(test)

    predict(test)


