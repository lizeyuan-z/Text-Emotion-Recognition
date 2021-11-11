import re

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from pytorch_metric_learning import losses

from dataset import MyDataset
from Model import BERT

BATCH_SIZE = 64
NUM_CLASSES = 9
seqs = []
labels = []
with open('data/SinaNews_processed.txt', 'r', encoding='utf-8') as f:
    for line in f:
        tmp = line.strip().split('\t')
        # seqs.append(re.sub(r'\s*', '', tmp[0]))
        seqs.append(tmp[0])
        labels.append(int(tmp[1]))

data = MyDataset(seqs, labels)
train_size = int(len(data) * 0.6)
test_size = len(data) - train_size
train_data, eval_data = torch.utils.data.random_split(data, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=BATCH_SIZE)


def train(model):
    for i, (seq, label) in enumerate(train_loader):
        out = model(seq)
        loss = cls_loss(out, label.to(device))
        loss.backward()
        loss_optimizer.step()
        print('>>>  batch:{0}     loss{1:.8f}'.format(i, loss))


def eval(model):
    labels_predicted = []
    labels_true = []
    for seq, label in eval_loader:
        out= model(seq)
        labels_predicted += out.max(1)[1].cpu().tolist()
        labels_true += list(label)
    f1 = f1_score(labels_true, labels_predicted, average='micro')
    print('F1-MICRO: {0:.5f}'.format(f1))
    return f1


config = 'bert-base-uncased'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
max_length = 50
model = BERT(config, device, max_length, NUM_CLASSES).to(device)
cls_loss = nn.CrossEntropyLoss().to(device)
loss_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
max_f1 = 0
for k in range(30):
    print("BATCH:", k)
    model.train()
    train(model)
    model.eval()
    f1 = eval(model)
    if f1 > max_f1:
        max_f1 = f1
        torch.save(model.state_dict(), 'SinaNews2012_')
