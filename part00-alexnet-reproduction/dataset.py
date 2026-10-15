from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(227),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = (0.4914, 0.4822, 0.4465),
        std = (1.0, 1.0, 1.0),
    ),
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(1.0, 1.0, 1.0),
    ),
])

train_dataset = datasets.CIFAR10(
    root = "./data",
    train = True,
    download = True,
    transform = train_transform,
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform,
)

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 64,
    shuffle = True,
    num_workers = 4,
    pin_memory = True,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

if __name__ == "__main__":
    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))

    print("训练集大小：", len(train_dataset))
    print("测试集大小：", len(test_dataset))

    print("训练 batch：", train_images.shape)
    print("测试 batch：", test_images.shape)

    print("训练标签：", train_labels[:10])
    print("测试标签：", test_labels[:10])