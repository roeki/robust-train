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
    learning_rate = 0.0001

    trainloader, testloader = get_cifar()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = Cifar_Small_ConvNet(
        non_negative=[False, False, False, False, False, False, False],
        norm=[False, False, False, False, False, False, False]
    )
    net.load_state_dict(torch.load('models/cifar_small_convnet_16eps_ours_2116.pth'))
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000,4000,6000,8000], gamma=0.4)
    eps = 16.7 / 255
    epoch = 0
    itr = 0
    k = 1
    best_verified_acc_flag = -1.0
    while itr < 10000:  # itr数除以250 即周期数
        # bs=200，itr=250一轮，遍历5w个样本需要个250itr--是一个epoch
        # bs=100 itr=500一轮
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
            temp = outputs  # [输出,输出]
            outputs = outputs[:outputs.shape[0] // 2]
            outputs_original = outputs  # [输出]

            x_ub = inputs + eps
            x_lb = inputs - eps
            outputs = net.forward(torch.cat([x_ub, x_lb], 0))
            z_hb = outputs[:outputs.shape[0] // 2]
            z_lb = outputs[outputs.shape[0] // 2:]
            lb_mask = torch.eye(10).cuda()[labels]
            hb_mask = 1 - lb_mask
            outputs = z_lb * lb_mask + z_hb * hb_mask
            _, predicted_veri = torch.max(outputs.data, 1)
            _, predicted_ori = torch.max(outputs_original.data, 1)

            if (predicted_ori == labels).sum().item() >= 165:
                loss += k * criterion(outputs, labels)
            else:
                if (predicted_ori == predicted_veri).sum().item() >= 375:
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs, predicted_ori)


                # loss += ((dev.pow(2).sum()) / (temp.pow(2).sum())).sqrt()
                # loss += k * (((z_hb - z_lb).pow(2).sum()).sqrt() + net.deviation_sqrt_sum(torch.cat([x_ub, x_lb], 0)))/2000
                # loss += (net.deviation(torch.cat([x_ub, x_lb], 0)) + ((z_hb - z_lb).pow(2)).sum())/80000
                # loss += k * (((z_hb - z_lb).pow(2).sum()).sqrt() + net.deviation_without(torch.cat([x_ub, x_lb], 0)))/500
                # print((temp + k * criterion(outputs, labels)) / (
                #             ((z_hb - z_lb).pow(2).sum()) + net.deviation(torch.cat([x_ub, x_lb], 0))))
                # print("loss:",loss)
                # print("dev:",net.deviation(torch.cat([x_ub, x_lb], 0)) + ((z_hb - z_lb).pow(2)).sum())
                # ((z_hb - z_lb).pow(2)).sum() 最后一层上界-下界 的平方的和
            # print(((z_hb - z_lb).pow(2).sum()).sqrt())
            # print("-----")
            # print(loss)
            loss.backward()
            optimizer.step()
            # print('{} scheduler: {}'.format(i, scheduler.get_last_lr()[0]))
            scheduler.step()
            itr += 1
            running_loss += loss.item()

        scheduler.step()
        # print(criterion(outputs, labels)/(net.deviation(torch.cat([x_ub, x_lb], 0)) + ((z_hb - z_lb).pow(2)).sum()))
        print('Epoch [%d, 320] loss: %.3f' % (epoch + 1, running_loss / 250))

        net.eval()
        print("Acc on Clean case:", print_accuracy(net, trainloader, testloader, device, test=True, eps=0))
        print("Acc on Worst case with", eps, "eps:",
              print_accuracy(net, trainloader, testloader, device, test=True, eps=eps))
        print("Acc on Worst case with 16/255 eps:",
              print_accuracy(net, trainloader, testloader, device, test=True, eps=16 / 255))
        epoch += 1
        if print_accuracy(net, trainloader, testloader, device, test=True, eps=16 / 255) > best_verified_acc_flag:
            best_verified_acc_flag = print_accuracy(net, trainloader, testloader, device, test=True, eps=16/ 255)
            torch.save(net.state_dict(), 'models/cifar_small_convnet_retrain.pth')


if __name__ == "__main__":
    app.run(main)
