import torchvision.datasets
from torch import nn

# train_data = torchvision.datasets.ImageNet(root='data/data_image_net', split='train',
#                                            download=True, transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print('OK')
# print(vgg16_true)

train_data = torchvision.datasets.CIFAR10('data/dataset', train=True,
                                          download=True, transform=torchvision.transforms.ToTensor())

vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
# print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
