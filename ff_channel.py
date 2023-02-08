# -*- coding: utf-8 -*-
"""FF_channel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1v6yLT8TRF_Kzfo7CrkG4HD1WJarKBosb

> ## binary版本的信道

每个bit的翻转率都是固定的，其实就是个普通的BSC，但是p我也没说，看你自己能不能学出来咯~

输入是一个np格式的{0,1}向量，输出是一个同样的向量
"""

import numpy as np

def Channel (x):
  p=0.2
  noise = np.random.choice([0, 1], size=x.shape, p=[1-p,p])
  return (x+noise)%2

Channel(np.array([1,0,1]))

"""> ## real number版本的信道

这是awgn信道，每个symbol会被叠加一个高斯噪声，方差=snr=Eb/No是未知的

输入是一个np格式的每两个相邻数都是一个模长<=1的点的向量，输出是同样的向量但模长可以大于1.

我们考虑的是BPSK
"""

import numpy as np

def Channel (x):
  No = 0.5 #假设Signal强度总是0.5,因为一个bpsk符号的能量一般是1，所以平均到每个bit就是0.5
  noise = np.random.randn(len(x))
  return x+noise*np.sqrt(No)

x = np.random.choice([0, 1], size=100, p=[0.5,0.5])
y = Channel(x)
np.sum((y-x)**2)/100