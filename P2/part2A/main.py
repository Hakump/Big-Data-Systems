import os
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import model as mdl
import time
import argparse

device = "cpu"
torch.set_num_threads(4)

batch_size = 256  # batch for one node

parser = argparse.ArgumentParser()
parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--num-nodes', '--num_nodes', default=4, type=int)
parser.add_argument('--master-ip', '--master_ip', default='10.10.1.1', type=str)
args = parser.parse_args()
def train_model(model, train_loader, optimizer, criterion, epoch, rank):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()  # ???
    total_time = 0.0
    total_loss = 0.0
    # remember to exit the train loop at end of the epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        # Your code goes here!
        data, target = data.to(device), target.to(device)
        begin = time.time()
        outputs = model(data)
        optimizer.zero_grad()
        loss = criterion(outputs, target)

        loss.backward()
        for param in model.parameters():
            gatherList = [torch.zeros_like(param.grad) for _ in range(args.num_nodes)]
            if rank == 0:
                torch.distributed.gather(tensor=param.grad, gather_list=gatherList)
                result_grad = torch.zeros_like(param.grad)
                for i in range(args.num_nodes):
                    result_grad += gatherList[i]
                torch.div(result_grad, args.num_nodes)
                for i in range(args.num_nodes):
                    gatherList[i] = result_grad
                torch.distributed.scatter(tensor=param.grad, scatter_list=gatherList)
            else:
                torch.distributed.gather(tensor=param.grad)
                torch.distributed.scatter(tensor=param.grad)

        optimizer.step()
        end = time.time()
        total_loss += loss.item()

        if batch_idx % 20 == 19:
            print("iteration: %5d loss: %.6f" % (batch_idx, total_loss))

        if 0 < batch_idx < 10:
            total_time += end - begin

    print('time: %.3f' % (total_time / 9))

    return None


def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():

    torch.manual_seed(1)
    torch.distributed.init_process_group(backend="gloo",  init_method= 'tcp://'+ args.master_ip + ':6585' , rank=args.rank, world_size=args.num_nodes)

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    training_set = datasets.CIFAR10(root="./data", train=True,
                                    download=True, transform=transform_train)
    training_sampler = torch.utils.data.distributed.DistributedSampler(training_set, num_replicas=args.num_nodes, rank=args.rank)
    train_loader = torch.utils.data.DataLoader(training_set,
                                               num_workers=2,
                                               batch_size=batch_size,
                                               sampler=training_sampler,
                                               shuffle=False,
                                               pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    # model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    # print(model.parameters())

    # running training for one epoch
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch, args.rank)
        test_model(model, test_loader, training_criterion)


if __name__ == "__main__":
    main()
