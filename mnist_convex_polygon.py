import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch.nn as nn
import math
import copy
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from network import *
from util import *

learning_rate = 0.001

trainloader, testloader = get_mnist()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = MNIST_Medium_ConvNet(
    non_negative=[False, False, False, False, False, False, False],
    norm=[False, False, False, False, False, False, False]
)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15000, 25000], gamma=0.1)
eps = 2 / 255 * 1.1
running_eps = 0
epoch = 0
itr = 0
while itr < 60000:
    running_loss = 0
    for i, data in enumerate(trainloader, 0):
        net.train()

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        loss = 0
        optimizer.zero_grad()

        outputs = net(torch.cat([inputs, inputs], 0))
        outputs = outputs[:outputs.shape[0] // 2]
        loss += criterion(outputs, labels)

        if itr > 2000 and itr < 12000:
            running_eps += eps / 10000

        if itr > 2000:
            x_ub = inputs + running_eps
            x_lb = inputs - running_eps
            outputs = net.forward_g(torch.cat([x_ub, x_lb], 0))
            v_hb = outputs[:outputs.shape[0] // 2]
            v_lb = outputs[outputs.shape[0] // 2:]
            weight = net.score_function.weight
            bias = net.score_function.bias
            w = weight.unsqueeze(0).expand(v_hb.shape[0], -1, -1) - weight[labels].unsqueeze(1)
            b = bias.unsqueeze(0).expand(v_hb.shape[0], -1) - bias[labels].unsqueeze(-1)
            u = ((v_hb + v_lb) / 2).unsqueeze(1)
            r = ((v_hb - v_lb) / 2).unsqueeze(1)
            w = torch.transpose(w, 1, 2)
            out_u = (u @ w).squeeze(1) + b
            out_r = (r @ torch.abs(w)).squeeze(1)
            loss += torch.mean(torch.log(1 + torch.exp(out_u + out_r)))

        loss.backward()
        optimizer.step()
        itr += 1
        running_loss += loss.item()
    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 600))
    net.eval()
    print_accuracy(net, trainloader, testloader, device, test=True, eps=0)
    if itr > 250000:
        print("verified test acc:", verify_robustness(net, testloader, device, eps=2 / 255))
    epoch += 1
