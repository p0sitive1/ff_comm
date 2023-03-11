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
            self.layers += [Layer(dims[d], dims[d + 1], id=d, device=device)]

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
            print(f"training layer {i}")
            h_pos, h_neg = layer.train(h_pos, h_neg)

class Layer(nn.Linear):
    def __init__(self, in_features, out_features, id,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.id = id
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 10000
        self.device = device

    def forward(self, x):
        x_direction = torch.nn.functional.normalize(x)
        x_direction = torch.tensor(x_direction, dtype=torch.float32)
        return self.relu(
            torch.matmul(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        if self.id == 2:
            for _ in tqdm(range(self.num_epochs)):
                g_pos = self.Channel(self.forward(x_pos))
                g_neg = self.Channel(self.forward(x_neg))
                g_pos_sum = g_pos[:, :128].pow(2) + g_pos[:, 128:].pow(2)
                g_neg_sum = g_neg[:, :128].pow(2) + g_neg[:, 128:].pow(2)
                g_pos_sum = torch.cat([g_pos_sum, g_pos_sum], 1)
                g_neg_sum = torch.cat([g_neg_sum, g_neg_sum], 1)
                g_pos = torch.div(g_pos, g_pos_sum)
                g_neg = torch.div(g_neg, g_neg_sum)
                loss = torch.relu(-(g_pos - g_neg).pow(2)).mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        else:
            for _ in tqdm(range(self.num_epochs)):
                g_pos = self.forward(x_pos)
                g_neg = self.forward(x_neg)
                g_pos_sum = g_pos[:, :128].pow(2) + g_pos[:, 128:].pow(2)
                g_neg_sum = g_neg[:, :128].pow(2) + g_neg[:, 128:].pow(2)
                g_pos_sum = torch.cat([g_pos_sum, g_pos_sum], 1)
                g_neg_sum = torch.cat([g_neg_sum, g_neg_sum], 1)
                g_pos = torch.div(g_pos, g_pos_sum)
                g_neg = torch.div(g_neg, g_neg_sum)
                loss = torch.relu(-(g_pos - g_neg).pow(2)).mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

    def Channel(self, x): 
        # add some noise
        stddev = np.sqrt(1 / (10 ** (6 / 10)))
        noise = torch.normal(mean=0, std=stddev, size=x.shape).to(device)
        return x + noise
    

class decoder(nn.Module):
    def __init__(self, block_length):
        super().__init__()
        self.conv1 = nn.Linear(block_length, 512)
        self.conv2 = nn.Linear(512, 256)
        self.conv3 = nn.Linear(256, 128)

        self.opt = Adam(self.parameters(), lr=0.03)
        self.cri = nn.MSELoss()
    
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(x)
        return x

def Channel(x): 
    # add some noise
    stddev = np.sqrt(1 / (10 ** (6 / 10)))
    noise = torch.normal(mean=0, std=stddev, size=x.shape).to(device)
    return x

def dec_to_bin(input, out_length=4):
    tmp = str(bin(input))[2:]
    if len(tmp) > out_length:
        raise("input length too short")
    if len(tmp) < out_length:
        tmp = (out_length - len(tmp)) * "0" + tmp
    output = list()
    for i in range(out_length):
        output.append(int(tmp[i]))
    return np.array(output)


if __name__ == "__main__":
    batch_size = 512
    block_length = 128

    train_data = np.random.binomial(1, 0.5, [10000, block_length])
    label_true, label_false = train_data, train_data
    np.random.shuffle(label_false)

    print(train_data.shape)
    print(label_true.shape)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    net_in = Net([block_length, 256, 256, 256], device)
    # net_out = Net([block_length, 256, 128, 64, 32, 1], device)

    x_pos = torch.Tensor(label_true).to(device)
    x_neg = torch.Tensor(label_false).to(device)
    
    net_in.train(x_pos, x_neg)

    imm_data = torch.nn.functional.normalize(net_in.forward(x_pos)).cpu().detach()
    imm_label = x_pos.cpu().detach()

    decode_train = torch.Tensor(np.concatenate((imm_data, imm_label), axis=1)).to(device)

    print(decode_train.shape)

    deco = decoder(256).to(device)
    for epoch in tqdm(range(10000)):
        data = decode_train[:, :256]
        label = decode_train[:, 256:]
        deco.opt.zero_grad()
        output = deco(data)
        loss = nn.functional.mse_loss(output, label)
        loss.backward()
        deco.opt.step()

        # print(f"current loss: {loss.item()}")

    test_data = np.random.binomial(1, 0.5, [10, block_length])
    test_label = test_data

    x_pos = torch.Tensor(test_data).to(device)

    encoded_msg = net_in.forward(x_pos)

    print(encoded_msg)

    channeled_msg = Channel(encoded_msg).to(device)

    print(channeled_msg)

    output = deco(channeled_msg)
    # output = torch.Tensor(np.where(output.cpu().detach().numpy() >= 0.5, 1, 0)).to(device)

    print(output)
    print(test_label)
    
    print(test_label - output.cpu().detach().numpy())
