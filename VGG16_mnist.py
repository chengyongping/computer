import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import copy
import matplotlib.pyplot as plt
import numpy as np

root = "mnist_data/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

dataset = {
    "train": datasets.MNIST(
        root=root,
        transform=transform,
        train=True,
        download=True
    ),
    "test": datasets.MNIST(
        root=root,
        transform=transform,
        train=False
    )
}

dataset_size = {x: len(dataset[x]) for x in ["train", "test"]}

data_loader = {
    x: DataLoader(
        dataset=dataset[x], batch_size=256, shuffle=True
    ) for x in ["train", "test"]
}


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(tensor=m.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=m.bias, val=0)

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
num_epochs = 100

net.to(device)


def train(net, optimizer, criterion, num_epochs=100, lr_reduce=5, early_stop=10, logger="./logger.txt"):
    best_wts = copy.deepcopy(net.state_dict())
    best_acc = 0
    cnt_for_lr_reduce = 0
    cnt_for_early_stop = 0

    for epoch in range(num_epochs):

        print("epoch {}/{}".format(epoch + 1, num_epochs))

        for phase in ["train", "test"]:

            running_loss = 0.0
            running_corrects = 0

            if phase == "train":
                net.train()
            else:
                net.eval()

            for inputs, labels in data_loader[phase]:

                optimizer.zero_grad()

                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    _, labels_hat = outputs.data.max(dim=1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                running_corrects += sum(labels_hat == labels.data).item()

            loss = running_loss / dataset_size[phase]
            acc = running_corrects / dataset_size[phase] * 100
            print("{} loss: {:.6f}, acc: {:.4f}%".format(phase, loss, acc))

            with open(logger, mode="a+") as f:
                f.write("{} epoch: {}, loss: {:.6f}, acc: {:.4f}%, lr: {:.6f}\n".format(
                    phase, epoch, loss, acc, optimizer.param_groups[0]["lr"]
                ))

            if phase == "test":

                if acc > best_acc:
                    best_acc = acc
                    best_wts = copy.deepcopy(net.state_dict())
                    torch.save(
                        net.state_dict(),
                        "best_net-epoch_{}-val_loss_{:.6f}-val_acc_{:.4f}.pth".format(
                            epoch, loss, acc
                        )
                    )
                    cnt_for_lr_reduce = 0
                    cnt_for_early_stop = 0
                else:
                    cnt_for_lr_reduce += 1
                    cnt_for_early_stop += 1

        # learning rate
        if cnt_for_lr_reduce > lr_reduce:
            for p in optimizer.param_groups:
                p['lr'] *= 0.9
            cnt_for_lr_reduce = 0
            print("lr reduced")

        if cnt_for_early_stop > early_stop:
            break

    torch.save(net.state_dict(), "last.pth")

train(net, optimizer, criterion, logger="adam_log.txt")


optimizer = optim.SGD(net.parameters(), lr=0.0001)
train(net, optimizer, criterion, logger="sgd_log.txt")




