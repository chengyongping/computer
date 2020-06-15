

import torchvision as tv
import torch.nn as nn
import torch as t
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim

MAX_EPOCH = 2
CLASS_NUM = 10


class Net(nn.Module):  # 定义网络
    def __init__(self, class_num=CLASS_NUM):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, class_num)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def getData():  # 数据预处理
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # 训练集
    train_set = tv.datasets.CIFAR10(root='/data/', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    # 测试集
    test_set = tv.datasets.CIFAR10(root='/data/', train=False, transform=transform, download=True)
    test_loader = t.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader, classes


def train():  # 训练
    net = Net()
    train_dataloader, test_dataloader, classes = getData()  # 加载数据
    ceterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(MAX_EPOCH):
        for step, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = ceterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if step % 3000 == 39:
                acc = test_net(net, test_dataloader)
                print('Epoch: ', epoch, ' |step: ', step, ' |train_loss: ', loss.item(),
                      '|test accuracy:%.4f' % acc)
        print('Finished Training')
        return net


def test_net(net, test_dataloader):  #  获取在测试集上的准确率
    correct, total = .0, .0
    for inputs, label in test_dataloader:
        output = net(inputs)
        _, predicted = t.max(output, 1)  #  获取分类结果
        total += label.size(0)  # 记录总个数
        correct += (predicted == label).sum()  #  记录分类正确的个数  
    return float(correct) / total


if __name__ == '__main__':
    net = train()







