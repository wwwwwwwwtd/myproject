import torchvision
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root='data',transform=torchvision.transforms.ToTensor(),download=True)
train_loader = DataLoader(train_data,batch_size=64,shuffle=True,drop_last=True)

    Conv2d(3, 32, 5, padding=2),
                MaxPool2d(2),
                Conv2d(32, 32, 5, padding=2),
                MaxPool2d(2),
                Conv2d(32, 64, 5, padding=2),
                MaxPool2d(2),
                Flatten(),
                Linear(1024, 64),
                Linear(64, 10)
