import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
# 准备训练集和测试集 ./是当前目录， ../是上一级目录
# 如果没有下载链接，点击ctrl转到CIFAR10进入查看url地址
train_set = torchvision.datasets.CIFAR10(root="./data/dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./data/dataset", train=False, transform=dataset_transform, download=True)

# print(test_set[0])
# # 查看数据的类别有什么
# print(test_set.classes)
#
# # 查看测试集的第一张图片是什么类别，查看图片
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

# print(test_set[0])

writer = SummaryWriter("p02")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()