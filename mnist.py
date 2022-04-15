import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from absl import app
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
import ssl
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

ssl._create_default_https_context = ssl._create_unverified_context
from network import *
from util import *


def main(_):
    num_class = 10

    train_sum = 60000
    ################ 参数设置调整区
    batchsize = 100
    learning_rate = 0.0005
    lr_decay = 0.1
    epoch_max = 300
    lr_scheduler1 = 130
    lr_scheduler2 = 250
    schedule_start = 10
    schedule_end = 60
    k_end = 1
    eps_test = 0.3
    eps_train = 0.4

    ################ 参数设置调整区
    itr_max = (train_sum / batchsize) * epoch_max
    lr_scheduler1 = int((train_sum / batchsize) * lr_scheduler1)
    lr_scheduler2 = int((train_sum / batchsize) * lr_scheduler2)
    trainloader, testloader = get_mnist(batch_size=batchsize)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = MNIST_Large_ConvNet(
        non_negative=[False, False, False, False, False, False, False],
        norm=[False, False, False, False, False, False, False]
    )
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_scheduler1,lr_scheduler2], gamma=lr_decay)

    running_eps = 0
    epoch = 0
    itr = 0
    k = 0
    best_verified_acc_flag = -1.0

    while itr < itr_max:  # itr数除以250 即周期数
        # bs=200，itr=250一轮，遍历5w个样本需要个250itr--是一个epoch
        # bs=10 itr=5000一轮
        # print(itr)

        running_loss = 0

        for i, data in enumerate(trainloader, 0):
            # print(data.type)
            # print(data.shape)
            net.train()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # print(inputs.size())
            loss = 0

            optimizer.zero_grad()

            outputs = net(torch.cat([inputs, inputs], 0))

            outputs = outputs[:outputs.shape[0] // 2]
            outputs_original = outputs  # [输出]

            loss += (1 - k) * criterion(outputs, labels)

            if (train_sum / batchsize) * schedule_start < itr <= (train_sum / batchsize) * schedule_end:
                running_eps += eps_train / ((train_sum / batchsize) * (schedule_end - schedule_start))
                k += k_end / ((train_sum / batchsize) * (schedule_end - schedule_start))

            if itr > (train_sum / batchsize) * schedule_start:
                x_ub = inputs + running_eps
                x_lb = inputs - running_eps
                outputs = net.forward(torch.cat([x_ub, x_lb], 0))
                z_hb = outputs[:outputs.shape[0] // 2]
                z_lb = outputs[outputs.shape[0] // 2:]
                lb_mask = torch.eye(10).cuda()[labels]
                hb_mask = 1 - lb_mask
                outputs = z_lb * lb_mask + z_hb * hb_mask


                loss += k * criterion(outputs, labels)


            loss.backward()
            optimizer.step()
            # print('{} scheduler: {}'.format(i, scheduler.get_last_lr()[0]))
            scheduler.step()
            itr += 1
            running_loss += loss.item()

        scheduler.step()
        # print(criterion(outputs, labels)/(net.deviation(torch.cat([x_ub, x_lb], 0)) + ((z_hb - z_lb).pow(2)).sum()))
        print('Epoch [%d, %d] loss: %.3f' % (epoch + 1, epoch_max, running_loss / 250))

        net.eval()
        print("Acc on Clean case:", print_accuracy(net, trainloader, testloader, device, test=True, eps=0))
        print("Acc on Worst case with", running_eps, "eps:",
              print_accuracy(net, trainloader, testloader, device, test=True, eps=running_eps))
        if itr > ((train_sum / batchsize) * schedule_start):
            print("Acc on Worst case with", eps_test, " eps:",
                  print_accuracy(net, trainloader, testloader, device, test=True, eps=eps_test))
        epoch += 1
        if print_accuracy(net, trainloader, testloader, device, test=True, eps=eps_test) > best_verified_acc_flag:
            best_verified_acc_flag = print_accuracy(net, trainloader, testloader, device, test=True, eps=eps_test)
            torch.save(net.state_dict(), 'models/mnist_large_convnet_03eps_traditional.pth')


if __name__ == "__main__":
    app.run(main)
