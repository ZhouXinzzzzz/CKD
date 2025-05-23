import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image


def get_data_folder():
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/places365")
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder


class Places365Instance(datasets.Places365):
    """Places365Instance Dataset."""

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


#  for CRD
class Places365InstanceSample(datasets.Places365):
    """
    Places365Instance+Sample Dataset
    """

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        k=4096,
        mode="exact",
        is_sample=True,
        percent=1.0,
    ):
        super().__init__(
            root=root,
            train=train,
            download=download,
            transform=transform,
            target_transform=target_transform,
        )
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 365
        num_samples = len(self.data) 
        label = self.targets  

        self.cls_positive = [[] for i in range(num_classes)] 
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)] 
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j]) 


        self.cls_positive = [
            np.asarray(self.cls_positive[i]) for i in range(num_classes)
        ]
        self.cls_negative = [
            np.asarray(self.cls_negative[i]) for i in range(num_classes)
        ]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [
                np.random.permutation(self.cls_negative[i])[0:n]
                for i in range(num_classes)
            ]

        self.cls_positive = np.asarray(self.cls_positive) 
        self.cls_negative = np.asarray(self.cls_negative) 


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            return img, target, index
        else:
            if self.mode == "exact":
                pos_idx = index
            elif self.mode == "relax":
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(
                self.cls_negative[target], self.k, replace=replace 
            )
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx)) 

            return img, target, index, sample_idx


def get_Places365_train_transform():
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return train_transform



def get_Places365_test_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )



def get_places365_dataloaders(batch_size, val_batch_size, num_workers):
    data_folder = get_data_folder()
    train_transform = get_Places365_train_transform()
    test_transform = get_Places365_test_transform()
    train_set = Places365Instance(
        root=data_folder, download=False, split='train-standard', small=True, transform=train_transform
    )
    num_data = len(train_set)
    test_set = datasets.Places365(
        root=data_folder, download=False, split='val', small=True, transform=test_transform
    ) 
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=1,
    )
    return train_loader, test_loader, num_data


# for CRD
def get_places365_dataloaders_sample(
    batch_size, val_batch_size, num_workers, k, mode="exact"
):
    data_folder = get_data_folder()
    train_transform = get_Places365_train_transform()
    test_transform = get_Places365_test_transform()

    train_set = Places365InstanceSample(
        root=data_folder,
        download=False,
        split='train-standard',
        small=True,
        transform=train_transform,
        k=k,
        mode=mode,
        is_sample=True,
        percent=1.0,
    )
    num_data = len(train_set)
    test_set = datasets.Places365(
        root=data_folder, download=False, split='val', small=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader, num_data





