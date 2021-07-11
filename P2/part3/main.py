import argparse
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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time


parser = argparse.ArgumentParser()
parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--master_ip', '--master-ip', default='10.10.1.1', type=str)
parser.add_argument('--num_nodes','--num-nodes', default=4, type=int)
args = parser.parse_args()
device = "cpu"
torch.set_num_threads(4)

batch_size = 256 # batch for one node
torch.manual_seed(1)
np.random.seed(1)


def setup():
    dist.init_process_group(backend='gloo', 
                            init_method= 'tcp://'+ args.master_ip + ':6585' , 
                            rank=int(args.rank), 
                            world_size=args.num_nodes)
    print('pid: ' +  str(os.getpid()), 'rank: '+  str(dist.get_rank()))



def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    totalTime = 0.0
    # remember to exit the train loop at end of the epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        begin = time.time()

        # Your code goes here!
        data, target = data.to(device), target.to(device)
        data_pred = model(data)

        loss = criterion(data_pred, target)
            
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        end = time.time()
        if (batch_idx > 0 and batch_idx < 10):
            totalTime += end - begin
        if(batch_idx % 20 == 0):
            print(batch_idx, loss.item())
    print('total time: ' + str(totalTime))
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
    setup()
    
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
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
    training_sampler = torch.utils.data.distributed.DistributedSampler(training_set, num_replicas=4, rank=int(args.rank))
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
    model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    # running training for one epoch
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        test_model(model, test_loader, training_criterion)
    # Tear down the process group
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
