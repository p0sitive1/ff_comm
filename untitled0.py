# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/194G6Sy2MeE0GOqK6u33lVft3UHl3VdD0
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
import numpy as np
import random

class Net(torch.nn.Module):
    def __init__(self, dims, device=None):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1], device=device).cuda()]

    def predict(self, x):
        goodness_per_label = []
        for label in range(16):
            h = x
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            h_pos, h_neg = layer.train(h_pos, h_neg)

class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 50000
        self.device = device

    def forward(self, x):
        if len(x.shape) == 1:
            x_direction = torch.nn.functional.normalize(x, dim=0)
            x_direction = x_direction.unsqueeze(0)
        else:
            x_direction = torch.nn.functional.normalize(x)
        x_direction = torch.tensor(x_direction, dtype=torch.float32).to(self.device)
        return self.relu(
            torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for _ in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

def Channel(x): 
    x = x.cpu().detach().numpy()
    return x

def get_pair(flag,label):
  label_true = label%16
  label_false = (label+random.randrange(1,16))%16

  label_bin = str(bin(label_true))[2:]
  if len(label_bin) < 4:
    label_bin = (4-len(label_bin))*'0'+label_bin

  if flag == 1:
    data = label_false
  else:
    data = label_true

  data_bin = str(bin(data))[2:]
  if len(data_bin) < 4:
    data_bin = (4-len(data_bin))*'0'+data_bin

  data = []
  label = []
  for i in range(4):
    data.append(float(data_bin[i]))
    label.append(float(label_bin[i]))

  return np.array(data), np.array(label)

'''
def get_pair(flag,label):

    tmp = np.array(label%16)
    if flag == 1:
      tmp = (tmp+random.randrange(1,16))%16

    bt = str(bin(tmp))[2:]
    if len(bt) < 4:
        bt = (4 - len(bt)) * "0" + bt



    o = list()
    for b in bt:
        judge1 = random.choice([True, False])
        judge2 = random.choice([-1, 1])
        if judge1:
            o.append(int(b) + judge2 * 0.2)
        else:
            o.append(int(b))

    tmp = np.array(label%16)

    bt = str(bin(tmp))[2:]
    if len(bt) < 4:
        bt = (4 - len(bt)) * "0" + bt
    out = list()
    for t in bt:
      out.append(int(t))

    return np.array(out), np.array(o)
'''

K = 8
N = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net_in = Net([K, N, N, N], device)
net_out = Net([N + 4, N, N, K], device)

data_length = 1000
x_pos, y_pos = list(), list()
x_neg, y_neg = list(), list()
for i in range(data_length):
    yp, xp = get_pair(0, i)
    yn, xn = get_pair(1, i)
    tmpxp = np.concatenate((yp, xp))
    tmpxn = np.concatenate((yn, xn))
    x_pos.append(tmpxp)
    y_pos.append(yp)
    x_neg.append(tmpxn)
    y_neg.append(yn)

x_pos = np.array(x_pos)
x_neg = np.array(x_neg)
y_pos = np.array(y_pos)
y_neg = np.array(y_neg) 

print(x_pos.shape, y_pos.shape)
print(x_neg.shape, y_neg.shape)

x_pos, x_neg = torch.from_numpy(np.array(x_pos)).to(device), torch.from_numpy(np.array(x_neg)).to(device)
net_in.train(x_pos, x_neg)

immp = Channel(net_in.forward(x_pos))
immn = Channel(net_in.forward(x_neg))

immp = np.concatenate((y_pos, immp), axis=1)
immn = np.concatenate((y_neg, immn), axis=1)

#print(immp.shape)

in_pos = torch.from_numpy(immp).to(device)
in_neg = torch.from_numpy(immn).to(device)

net_out.train(in_pos, in_neg)

testl, testx = get_pair(0, 12)

test = np.concatenate((testl, testx))
test = torch.from_numpy(np.array(test)).float().to(device)

tmp = net_in.forward(test)
tmp = Channel(tmp)

ttmp = np.concatenate((testl, np.squeeze(tmp)))
tttmp = torch.from_numpy(ttmp).to(device)
tttmp = net_out.forward(tttmp)
print(tttmp)

tot = 0
err = 0
t_three = 0
print("testing")
for i in tqdm(range(100)):
    # =============??????================#
    # ?????????????????????data???label
    testl, testx = get_pair(0, i)
    # print(testl, testx)

    print(testl)

    # ???data???label concatenate

    test = np.concatenate((testl, testx))
    test = torch.from_numpy(np.array(test)).float().to(device)

    tmp = net_in.forward(test)

    imm = Channel(tmp)

    outputs = list()
    # ???????????????data????????????label concatenate??????
    for i in range(16):
        cur_l = str(bin(i))[2:]
        if len(cur_l) < 4:
            cur_l = (4 - len(cur_l)) * "0" + cur_l
        testlabel = list()
        for t in cur_l:
            testlabel.append(int(t))
        testlabel = np.array(testlabel)

        test = np.concatenate((testlabel, np.squeeze(imm)))

        test = torch.from_numpy(np.array(test)).to(device)

        out = net_out.forward(test)

        outputs.append(out)
    
    ground_truth = "0b"
    for l in testl:
        ground_truth += str(int(l))

    ground_truth = int(ground_truth, 2)
    # print(ground_truth)
    guesses = list()
    for v in range(len(outputs)):
        guesses.append([v, torch.sum(outputs[v]).item()])

    # print(guesses)
    guesses.sort(key=lambda x: x[1], reverse=True)
    # print(guesses)
    guess = guesses[0][0]

    for i in range(3):
        if int(ground_truth) == guesses[i][0]:
            t_three += 1

    if int(ground_truth) == int(guess):
        tot += 1
    else:
        err += 1
        tot += 1

print(f"test error: {err/tot}")
print(f"top three rate: {t_three/tot}")