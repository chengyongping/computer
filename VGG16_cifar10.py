import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import os
#from torchsummary import summary

# 预设参数
CLASS_NUM = 10
BATCH_SIZE = 128
EPOCH =25

# 检验GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# ----------------------------------------------------------------------------------------------------------------------
class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        # 输入shape 3*32*32
        self.conv1 = nn.Conv2d(3,64,3,padding=1)        # 64*32*32
        self.conv2 = nn.Conv2d(64,64,3,padding=1)       # 64*32*32
        self.pool1 = nn.MaxPool2d(2, 2)                 # 64*16*16
        self.bn1 = nn.BatchNorm2d(64)                   # 64*16*16
        self.relu1 = nn.ReLU()                          # 64*16*16

        self.conv3 = nn.Conv2d(64,128,3,padding=1)      # 128*16*16
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)   # 128*16*16
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)      # 128*9*9
        self.bn2 = nn.BatchNorm2d(128)                  # 128*9*9
        self.relu2 = nn.ReLU()                          # 128*9*9

        self.conv5 = nn.Conv2d(128,128, 3,padding=1)    # 128*9*9
        self.conv6 = nn.Conv2d(128, 128, 3,padding=1)   # 128*9*9
        self.conv7 = nn.Conv2d(128, 128, 1,padding=1)   # 128*11*11
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)      # 128*6*6
        self.bn3 = nn.BatchNorm2d(128)                  # 128*6*6
        self.relu3 = nn.ReLU()                          # 128*6*6

        self.conv8 = nn.Conv2d(128, 256, 3,padding=1)   # 256*6*6
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)  # 256*6*6
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1) # 256*8*8
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)      # 256*5*5
        self.bn4 = nn.BatchNorm2d(256)                  # 256*5*5
        self.relu4 = nn.ReLU()                          # 256*5*5

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1) # 512*5*5
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1) # 512*5*5
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1) # 512*7*7
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)      # 512*4*4
        self.bn5 = nn.BatchNorm2d(512)                  # 512*4*4
        self.relu5 = nn.ReLU()                          # 512*4*4

        self.fc14 = nn.Linear(512*4*4,1024)             # 1*1024
        self.drop1 = nn.Dropout2d()                     # 1*1024
        self.fc15 = nn.Linear(1024,1024)                # 1*1024
        self.drop2 = nn.Dropout2d()                     # 1*1024
        self.fc16 = nn.Linear(1024,10)                  # 1*10

    def forward(self,x):
        x = x.to(device)  # 自加
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)


        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(-1,512*4*4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return x

# ----------------------------------------------------------------------------------------------------------------------

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# 从源文件读取数据
# 返回 train_data[50000,3072]和labels[50000]
#    test_data[10000,3072]和labels[10000]
def get_data(train=False):
    data = None
    labels = None
    if train == True:
        for i in range(1, 6):
            batch = unpickle('data/cifar-10-batches-py/data_batch_' + str(i))
            if i == 1:
                data = batch[b'data']
            else:
                data = np.concatenate([data, batch[b'data']])

            if i == 1:
                labels = batch[b'labels']
            else:
                labels = np.concatenate([labels, batch[b'labels']])
    else:
        batch = unpickle('data/cifar-10-batches-py/test_batch')
        data = batch[b'data']
        labels = batch[b'labels']
    return data, labels


# 图像预处理函数，Compose会将多个transform操作包在一起
# 对于彩色图像，色彩通道不存在平稳特性
transform = transforms.Compose([
    # ToTensor是指把PIL.Image(RGB) 或者numpy.ndarray(H * W * C)
    # 从0到255的值映射到0到1的范围内，并转化成Tensor格式。
    transforms.ToTensor(),
    # Normalize函数将图像数据归一化到[-1,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
)

# 将标签转换为torch.LongTensor
def target_transform(label):
    label = np.array(label)         # 变为ndarray
    target = torch.from_numpy(label).long()     # 变为torch.LongTensor
    return target


'''
自定义数据集读取框架来载入cifar10数据集
需要继承data.Dataset
'''

# 数据集
class Cifar10_Dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        # 初始化文件路径
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        # 载入训练数据集
        if self.train:
            self.train_data, self.train_labels = get_data(train)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            # 将图像数据格式转换为[height,width,channels]方便预处理
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
            # 载入测试数据集
        else:
            self.test_data, self.test_labels = get_data()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))
        pass

    # 从数据集中读取一个数据并对数据进行预处理返回一个数据对，如（data,label）
    def __getitem__(self, index):
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.test_data[index], self.test_labels[index]
        img = Image.fromarray(img)
        # 图像预处理
        if self.transform is not None:
            img = self.transform(img)
        # 标签预处理
        if self.target_transform is not None:
            target = self.target_transform(label)

        return img, target

    def __len__(self):
        # 返回数据集的size
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


if __name__ == '__main__':
    # 读取训练集和测试集
    train_data = Cifar10_Dataset(True, transform, target_transform)
    print('size of train_data:{}'.format(train_data.__len__()))
    test_data = Cifar10_Dataset(False, transform, target_transform)
    print('size of test_data:{}'.format(test_data.__len__()))
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    net = Net()
    net.to(device)
    # ------------------------------------------------
    with torch.no_grad():
        for input_data, _ in train_loader:
            break
        # summary(model.to(hyperparams['device']), input.size()[1:], device=hyperparams['device'])
        # print(input_data.size())
        #summary(net, input_data.size()[1:])
    os.system('pause')
    # ------------------------------------------------

    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    # 定义损失函数
    # 在使用CrossEntropyLoss时target直接使用类别索引，不适用one-hot
    loss_fn = nn.CrossEntropyLoss()

    loss_list = []
    Accuracy = []
    for epoch in range(1, EPOCH + 1):
        # 训练部分
        timestart = time.time()         # 自加计时
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(y)
            output = net(b_x)
            b_x, b_y = b_x.to(device), b_y.to(device)   # CPU 转 GPU
            loss = loss_fn(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录loss
            if step % 50 == 0:
                loss_list.append(loss)
        # 每完成一个epoch进行一次测试观察效果
        pre_correct = 0.0
        test_loader = Data.DataLoader(dataset=test_data, batch_size=100, shuffle=True)
        for (x, y) in (test_loader):
            b_x = Variable(x)
            b_y = Variable(y)
            b_x, b_y = b_x.to(device), b_y.to(device)       # 自加
            output = net(b_x)
            pre = torch.max(output, 1)[1]
            # pre_correct = pre_correct.to(device)        # 自加
            pre_correct = pre_correct + float(torch.sum(pre == b_y))
        print('EPOCH:{epoch},ACC:{acc}%'.format(epoch=epoch, acc=(pre_correct / float(10000)) * 100))
        Accuracy.append(pre_correct / float(10000) * 100)

        # 自加计时
        print('epoch %d cost %3f sec' % (epoch, time.time() - timestart))

    # 保存网络模型
    torch.save(net, 'lenet_cifar_10.model')
    # 绘制loss变化曲线
    plt.figure()
    plt.plot(loss_list)
    plt.figure()
    plt.plot(Accuracy)
    plt.show()

