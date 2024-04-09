import torchvision
import torch
import torchvision.transforms as transforms
import config as CFG


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def get_original_cifar10_dataloaders(
    root, train_bsz=256, val_bsz=256, img_size=CFG.size
):
    # Add resize if needed
    transform_train = transforms.Compose([
        # NOTE: Images are square, so resize one dim is sufficient
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])

    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bsz,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                        download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=val_bsz,
                                            shuffle=False, num_workers=2)

    return trainloader, testloader


def get_original_cifar100_dataloaders(
    root, train_bsz=256, val_bsz=256, img_size=CFG.size
):
    # Add resize if needed
    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])

    trainset = torchvision.datasets.CIFAR100(root=root, train=True,
                                            download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bsz,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root=root, train=False,
                                        download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=val_bsz,
                                            shuffle=False, num_workers=2)

    return trainloader, testloader


def get_imagenet_dataloaders(
    root, train_bsz=32, val_bsz=32, img_size=CFG.size, return_imgs=False
):
    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomResizedCrop([img_size, img_size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    #return_imgs = True
    if return_imgs:
        transform_train = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor()
        ])

    trainset = torchvision.datasets.ImageFolder(
        root=root+"/train", transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bsz,
                                            shuffle=True, num_workers=2)

    if return_imgs:
        testset = ImageFolderWithPaths(
            root=root+"/val", transform=transform_test
        )
    else:
        testset = torchvision.datasets.ImageFolder(
            root=root+"/val", transform=transform_test
        )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=val_bsz,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    return trainloader, testloader
