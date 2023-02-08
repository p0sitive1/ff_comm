import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import numpy as np
import random


def overlay_y_on_x(x, y):
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class Net(torch.nn.Module):
    def __init__(self, dims, device=None):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1], device=device).cuda()]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            # h = overlay_y_on_x(x, label)
            h = x
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)
    
    def forward(self, x):
        """
        forward x过整个网络
        """
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
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
        # x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
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
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


def Channel(x): 
    # clip data to [0, 1]
    # x = torch.nn.functional.normalize(x)
    x = x.cpu().detach().numpy()
    tmp = np.zeros(x.shape)
    v = np.max(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] > 0:
                tmp[i][j] = 1
            else:
                tmp[i][j] = 0

    # print("---x---")
    # print(x)
    # print("---t---")
    # print(tmp)
    x = tmp
    p = 0.2
    noise = np.random.choice([0, 1], size=x.shape, p=[1-p,p])
    out = (x + noise) % 2
    return out


def get_pair(flag=0, typ=0):
    """
    Flag=0 for positive data, flag=1 for negative data
    typ=0 for label in decimal, typ=1 for label in binary 
    """
    tmp = random.randrange(0, 16)
    bt = str(bin(tmp))[2:]
    if len(bt) < 4:
        bt = (4 - len(bt)) * "0" + bt

    if flag == 0:
        label = tmp
    else:
        label = random.randrange(0, 16)
        while label == tmp:
            label = random.randrange(0, 16)

    o = list()
    for b in bt:
        judge1 = random.choice([True, False])
        judge2 = random.choice([-1, 1])
        if judge1:
            o.append(int(b) + judge2 * 0.2)
        else:
            o.append(int(b))

    if typ == 1:
        tmpl = str(bin(label))[2:]
        if len(tmpl) < 4:
            tmpl = (4 - len(tmpl)) * "0" + tmpl
        out = list()
        for t in tmpl:
            out.append(int(t))
        label = np.array(out)

    return label, np.array(o)


def main():
    K = 8
    N = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_in = Net([K, N, N], device)
    net_out = Net([N + 4, N, K], device)

    # 生成数据
    data_length = 10000
    x_pos, y_pos = list(), list()
    x_neg, y_neg = list(), list()
    for i in range(data_length):
        xp, yp = get_pair(0, 1)
        xn, yn = get_pair(1, 1)
        tmpxp = np.concatenate((yp, xp))
        tmpxn = np.concatenate((yn, xn))
        x_pos.append(tmpxp)
        y_pos.append(yp)
        x_neg.append(tmpxn)
        y_neg.append(yn)

    label_pos = y_pos
    label_neg = y_neg
    x_pos, y_pos = torch.from_numpy(np.array(x_pos)).to(device), torch.from_numpy(np.array(y_pos)).to(device)
    x_neg, y_neg = torch.from_numpy(np.array(x_neg)).to(device), torch.from_numpy(np.array(y_neg)).to(device)

    # 训练前半段
    net_in.train(x_pos, x_neg)

    # 计算前半个网络的输出
    tmpp = np.concatenate((label_pos, Channel(net_in.forward(x_pos))), axis=1)
    tmpn = np.concatenate((label_neg, Channel(net_in.forward(x_neg))), axis=1)

    print(tmpp.shape)
    in_pos = torch.from_numpy(tmpp).to(device)
    in_neg = torch.from_numpy(tmpn).to(device)

    # 训练后半段
    net_out.train(in_pos, in_neg)


    tot = 0
    err = 0
    print("testing")
    for i in tqdm(range(100)):
        # =============测试================#
        # 获取一个随机的data和label
        testl, testx = get_pair(0, 1)
        # print(testx, testl)

        # 把data和label concatenate

        test = np.concatenate((testl, testx))
        test = torch.from_numpy(np.array(test)).float().to(device)

        tmp = net_in.forward(test)

        imm = torch.from_numpy(Channel(tmp)).to(device)

        tmp = imm.detach().cpu().numpy()

        # 把编码后的data和所有的label concatenate起来
        labels = list()
        data = list()
        for i in range(16):
            tmpl = str(bin(i))[2:]
            if len(tmpl) < 4:
                tmpl = (4 - len(tmpl)) * "0" + tmpl
            out = list()
            for t in tmpl:
                out.append(int(t))
            labels.append(np.array(out))
            data.append(tmp)

        test = list()
        for i in range(16):
            tmp = np.concatenate((labels[i], np.squeeze(data[i])))
            test.append(tmp)

        test = torch.from_numpy(np.array(test)).float().to(device)

        out = net_out.forward(test)

        # for o in range(len(out)):
        #     print(o, sum(out[o]).detach().item())
        
        ground_truth = "0b"
        for l in testl:
            ground_truth += str(l)

        ground_truth = int(ground_truth, 2)
        # print(ground_truth)
        output = dict()
        for o in range(len(out)):
            output[sum(out[o]).detach().item()] = o

        guess = output[max(output.keys())]
        # print(guess)

        if int(ground_truth) == int(guess):
            tot += 1
        else:
            err += 1
            tot += 1

    print(f"test error: {err/tot}")


if __name__ == "__main__":
    main()