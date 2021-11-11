import re

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from pytorch_metric_learning import losses

from dataset import MyDataset
from Model import BERT

BATCH_SIZE = 64
NUM_CLASSES = 8
seqs = []
labels = []
with open('data/ISEAR_processed.txt', 'r', encoding='utf-8') as f:
    for line in f:
        tmp = line.strip().split('\t')
        # seqs.append(re.sub(r'\s*', '', tmp[0]))
        seqs.append(tmp[0])
        labels.append(int(tmp[1]))

data = MyDataset(seqs, labels)
train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
train_data, eval_data = torch.utils.data.random_split(data, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=BATCH_SIZE)


def train(model):
    for i, (seq, label) in enumerate(train_loader):
        out = model(seq)
        loss1 = cls_loss(out, label.to(device))
        loss2 = cosface_loss(out, label)
        for l in range(len(label)):
            sim[label[l]] = sim[label[l]] + out[l]
        loss = loss1 + 0.001 * loss2
        loss.backward()
        loss_optimizer.step()
        # print('>>>  batch:{0}     loss{1:.8f}   cls_loss:{2:.8f}    cosface_loss:{3:.8f}'.format(i, loss, loss1, loss2))



def eval(model):
    labels_predicted = []
    labels_true = []
    for seq, label in eval_loader:
        out= model(seq)
        for l in range(len(label)):
            sim[label[l]] = sim[label[l]] + out[l]
        labels_predicted += out.max(1)[1].cpu().tolist()
        labels_true += list(label)
    f1 = f1_score(labels_true, labels_predicted, average='micro')
    print('F1-MICRO: {0:.5f}'.format(f1))
    sim = sim / len(labels)
    return f1


config = 'bert-base-uncased'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
max_length = 50
model = BERT(config, device, max_length, NUM_CLASSES).to(device)
cls_loss = nn.CrossEntropyLoss().to(device)
cosface_loss = losses.CosFaceLoss(NUM_CLASSES, NUM_CLASSES, margin=0.25, scale=16).to(device)
loss_optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 2e-6},
                  {'params': cosface_loss.parameters()}], lr=2e-5)
max_f1 = 0
for k in range(20):
    print("BATCH:", k)
    sim = torch.zeros(8, 8)
    model.train()
    train(model)
    model.eval()
    f1 = eval(model)
    print("corr:\n", sim @ sim.t())
    if f1 > max_f1:
        max_f1 = f1
        torch.save(model.state_dict(), 'ISEAR')
