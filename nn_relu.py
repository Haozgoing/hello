import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("data/dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        # output = self.relu1(input)
        output = self.sigmoid1(input)
        return output

cnn = CNN()
# output = cnn(input)
# print(output)

step = 0
writer = SummaryWriter('logs')
for data in dataloader:
    imgs, targets = data
    writer.add_images("sigmoidinput", imgs, step)
    output = cnn(imgs)
    writer.add_images("sigmoidoutput", output, step)
    step += 1
writer.close()
