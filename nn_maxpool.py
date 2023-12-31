import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("data/dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
#
# input = torch.reshape(input, (-1, 1, 5, 5) )

# print(input)
# print(input.shape)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

cnn = CNN()
# output = cnn(input)
# print(output)

step = 0
writer = SummaryWriter('logs_maxpool')
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = cnn(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()


