import torch.optim
import torchvision.datasets
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("data/dataset", train=False,
                                       download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


cnn = CNN()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(cnn.parameters(), lr=0.01)

# 迭代训练
num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = cnn(imgs)
        result_loss = loss(outputs, targets)
        # print(outputs)
        # print(targets)
        # print(result_loss)
        # 反向传播
        optim.zero_grad() # 清零梯度
        result_loss.backward() # 计算梯度
        # 参数更新
        optim.step()
        # print(result_loss)
        # 每一轮学习过程所有数据的误差总和loss
        running_loss = running_loss + result_loss
    print(running_loss)