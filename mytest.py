import copy
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from absl import app
from torch.nn.parameter import Parameter
from network import *
from util import *
def main(_):
    trainloader, testloader = get_mnist()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MNIST_Small_ConvNet(
        non_negative=[False, False, False, False,False,False,False],
        norm=[False, False, False,False,False,False, False]
    )
    net = net.to(device)
    net.load_state_dict(torch.load('./models/mnist_small_convnet_03eps_ours.pth'))

    print("clean acc:")
    print(print_accuracy(net, trainloader, testloader, device, test=True, eps=0))
    print("acc on Worst case with 0.3 eps")
    print(print_accuracy(net, trainloader, testloader, device, test=True, eps=0.3))

if __name__ == "__main__":
    app.run(main)