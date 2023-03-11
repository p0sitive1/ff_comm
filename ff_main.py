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
            print(f"training layer {i}")
            h_pos, h_neg = layer.train(h_pos, h_neg)

class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 100000
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

# def Channel(x): 
#     x = x.cpu().detach().numpy()
#     # add some noise
#     t = np.zeros(x.shape)
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             noise = random.choice([0.01, 0, -0.01])
#             t[i][j] = x[i][j] + noise
#     return t

def Channel(x):
    x = x.cpu().detach().numpy()
    stddev = np.sqrt(1 / (10 ** (6)))
    noise = torch.normal(mean=0, std=stddev, size=x.shape).to(device)
    noise = noise.cpu().detach().numpy()
    return x + noise

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

def generate_data(length):
    """
    generate training data, returns true label, false label, data
    """
    data_true = list()
    data_false = list()
    data_value = list()
    for _ in range(length):
        value = random.randrange(0, 16)
        label_true = value
        label_false = random.randrange(0, 16)
        while label_false == value:
            label_false = random.randrange(0, 16)

        value = dec_to_bin(value)
        label_true = dec_to_bin(label_true)
        label_false = dec_to_bin(label_false)

        data_true.append(label_true)
        data_false.append(label_false)
        data_value.append(value)
    return np.array(data_true), np.array(data_false), np.array(data_value)

def generate_spec_data(label, data):
    bi_label = dec_to_bin(label)
    bi_data = dec_to_bin(data)

    return np.array(bi_label), np.array(bi_data)

if __name__ == "__main__":
    label_true, label_false, data = generate_data(50000)

    K = 8
    N = 24

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net_in = Net([K, 32, 14], device)
    net_out = Net([14 + 4, 32, K], device)

    # training here
    x_pos = np.concatenate((label_true, data), axis=1)
    x_neg = np.concatenate((label_false, data), axis=1)

    x_pos = torch.Tensor(x_pos).to(device)
    x_neg = torch.Tensor(x_neg).to(device)
    
    net_in.train(x_pos, x_neg)

    imm_pos = Channel(net_in.forward(x_pos))
    imm_neg = Channel(net_in.forward(x_pos))

    imm_pos = np.concatenate((label_true, imm_pos), axis=1)
    imm_neg = np.concatenate((label_false, imm_neg), axis=1)

    imm_pos = torch.Tensor(np.array(imm_pos)).to(device)
    imm_neg = torch.Tensor(np.array(imm_neg)).to(device)

    net_out.train(imm_pos, imm_neg)

    # testing here
    tot = 0
    err = 0
    t_three = 0
    print("testing")
    for _ in range(100):
        label_true, label_false, data = generate_data(1)
        input = np.concatenate((label_true, data), axis=1)
        input = torch.Tensor(np.array(input)).to(device)

        imm = Channel(net_in.forward(input))

        outputs = list()
        for i in range(16):
            tmp = np.array([dec_to_bin(i)])

            cur_imm = np.concatenate((tmp, imm), axis=1)

            cur_imm = torch.Tensor(np.array(cur_imm)).to(device)

            output = net_out.forward(cur_imm)

            outputs.append(output)
        
        label_true = label_true.tolist()[0]
        for i in range(4):
            label_true[i] = str(label_true[i])
        ground_truth = "0b" + "".join(label_true)
        ground_truth = int(ground_truth, 2)

        guesses = list()
        for v in range(len(outputs)):
            guesses.append([v, torch.sum(outputs[v]).item()])
        guesses.sort(key=lambda x: x[1], reverse=True)
        guess = guesses[0][0]

        # print(ground_truth)
        # print(guesses)

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
