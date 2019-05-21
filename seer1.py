from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import os
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
def get_train_valid_loader(
        batch_size,
        random_seed,
        valid_size=0.1,
        shuffle=True,
        num_workers=4,
        pin_memory=False):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = datasets.CIFAR100(
        root='./data', train=True,
        download=True, transform=train_transform,
    )
    valid_dataset = datasets.CIFAR100(
        root='./data', train=True,
        download=True, transform=valid_transform,
    )
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return (train_loader, valid_loader)
def get_test_loader(
        batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    dataset = datasets.CIFAR100(
        root='./data', train=False,
        download=True, transform=transform,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return data_loader
size = np.random.randint(16,64)
class ConvNet(nn.Module):
    def __init__(self, dl, tl, ll): #dl: [iterations, channels/iter]
        self.numClasses = 100
        super(ConvNet, self).__init__()
        self.trlayer1 = nn.Sequential(
            nn.Conv2d(3, tl[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(tl[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer1 = []
        for i in range(dl[0][0]):
            self.layer1.append(nn.Sequential(
                nn.Conv2d(tl[0]+dl[0][1]*i, dl[0][1], kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(dl[0][1]),
                nn.ReLU()
                ))
        self.trlayer2 = nn.Sequential(
            nn.Conv2d(tl[0]+dl[0][0]*dl[0][1], tl[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(tl[1]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
            )
        self.layer2 = []
        for i in range(dl[1][0]):
            self.layer2.append(nn.Sequential(
                nn.Conv2d(tl[1]+dl[1][1]*i, dl[1][1], kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(dl[1][1]),
                nn.ReLU()
                ))
        self.trlayer3 = nn.Sequential(
            nn.Conv2d(tl[1]+dl[1][0]*dl[1][1], tl[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(tl[2]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
            )
        self.fc1 = nn.Linear(4*4*tl[2], ll[0])
        self.fc2 = nn.Linear(ll[0], 100)
    def forward(self, x):
        out = self.trlayer1(x)
        for i in range(8):
            aut = self.layer1[i](out)
            out = torch.cat((out, aut), 1)
        out = self.trlayer2(out)
        for i in range(8):
            aut = self.layer2[i](out)
            out = torch.cat((out, aut), 1)
        out = self.trlayer3(out)
#        out = self.trlayer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out
model = ConvNet([[8, 8], [8, 8]], [32, 32, 16], [128]).to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('/Users/nathansun/Desktop/python/quadratic.py')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
train_loader, valid_loader = get_train_valid_loader(
        128,
        np.random.seed(0),
        valid_size=0.1,
        shuffle=True,
        num_workers=4,
        pin_memory=False)
learning_rate=np.random.uniform(0.1,0.00004)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
def validate(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(batch_idx, len(valid_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, '/Users/nathansun/Desktop/python/quadratic.py')
        best_acc = acc
test_loader = get_test_loader(
        128,
        shuffle=True,
        num_workers=4,
        pin_memory=False)
def test():
    correct =0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            print("Acc:" , 100. * correct / total)
    print('Accuracy: %d %%' % (
            100 * correct / total))




# Number of epochs:
x=10




for epoch in range(start_epoch, start_epoch+x):
    print("learning rate:", learning_rate,"channel size: ", size)
    train(epoch)
    validate(epoch)
    learning_rate = np.random.uniform(0.1, 0.00004)
    size = np.random.randint(16, 64)
    #batch = np.random.randint(100, 200)
test() #final acc: 37% with CIFAR 100, 70% CIFAR 10 (old model)
