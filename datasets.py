import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST

CF_PAIRS = {
    'mnist': [(4, 9), (3, 8), (5, 6)],
    'fmnist': [(0, 2), (7, 9), (4, 6)]
}


def init_dataloader(args, is_train=True):
    """
        returns train and test dataloaders for a pair of classes.

    :param args:
    :param is_train:
    :return:
    """
    return dataloader_pair(args, is_train)


def dataloader_pair(args, is_train):
    """
        returns dataloaders for a class pair.

    :param args:
    :param is_train:
    :return:
    """

    dataset_1 = return_dataset(args, is_train)
    dataset_2 = return_dataset(args, is_train)

    # extract samples only of class 1
    idx_1 = dataset_1.targets == args.cls_1
    dataset_1.data = dataset_1.data[idx_1]
    dataset_1.targets = dataset_1.targets[idx_1]

    # extract samples only of class 2
    idx_2 = dataset_2.targets == args.cls_2
    dataset_2.data = dataset_2.data[idx_2]
    dataset_2.targets = dataset_2.targets[idx_2]

    return dataloader(dataset_1, is_train, args), dataloader(dataset_2, is_train, args)


def return_dataset(args, is_train):
    """
        returns an MNIST/FMNIST dataset

    :param args:
    :param is_train:
    :return:
    """
    if args.dataset == "mnist":
        dataset = MNIST(
                args.dataset_path,
                train=is_train,
                download=True,
                transform=return_transforms(args),
            )
    elif args.dataset == "fmnist":
        dataset = FashionMNIST(
                args.dataset_path,
                train=is_train,
                download=True,
                transform=return_transforms(args),
            )
    else:
        raise ValueError
    return dataset


def return_transforms(args):
    """
        returns a set of desired transformations.
    :param args:
    :return:
    """
    return transforms.Compose(
                [transforms.Resize(args.img_size),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5])
                 ]
            )


def dataloader(dataset, shuffle, args):
    """
        returns a dataloader for a given dataset.

    :param dataset:
    :param shuffle:
    :param args:
    :return:
    """
    return DataLoader(dataset,
                      batch_size=args.batch_size,
                      num_workers=args.n_cpu,
                      shuffle=shuffle,
                      pin_memory=True,
                      drop_last=False
                      )
