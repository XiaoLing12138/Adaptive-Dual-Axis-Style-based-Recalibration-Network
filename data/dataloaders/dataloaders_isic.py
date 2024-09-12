import os
import random

import numpy as np
import pandas as pd

from PIL import Image
from PIL import ImageFilter
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class ISICDataset(Dataset):
    def __init__(self, root_path, transform, mode):
        super(ISICDataset, self).__init__()
        self.root = root_path
        self.mode = mode
        if mode != "test":
            label_list = pd.read_csv(
                os.path.join(self.root,
                             "ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")
            )
        else:
            label_list = pd.read_csv(
                os.path.join(self.root,
                             "ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv")
            )

        # label
        self.labeled_imgs = label_list["image"].values
        temp_gr = label_list.iloc[:, 1:-1].values.astype(int).tolist()

        for index in range(len(temp_gr)):
            if temp_gr[index][0] == 1:
                temp_gr[index] = 0
            elif temp_gr[index][1] == 1:
                temp_gr[index] = 1
            elif temp_gr[index][2] == 1:
                temp_gr[index] = 2
            elif temp_gr[index][3] == 1:
                temp_gr[index] = 3
            elif temp_gr[index][4] == 1:
                temp_gr[index] = 4
            elif temp_gr[index][5] == 1:
                temp_gr[index] = 5
            else:
                temp_gr[index] = 6

        self.labeled_gr = np.array(temp_gr).astype(int)
        self.transform = transform

    def __getitem__(self, item):
        img_path = self.labeled_imgs[item]
        if self.mode != "test":
            img_path = os.path.join(self.root, "ISIC2018_Task3_Training_Input", img_path + ".jpg")
        else:
            img_path = os.path.join(self.root, "ISIC2018_Task3_Validation_Input", img_path + ".jpg")

        target = self.labeled_gr[item]

        img = Image.fromarray(io.imread(img_path)).convert("RGB")
        img = self.transform(img)
        return img, target.astype(int)

    def __len__(self):
        return self.labeled_imgs.shape[0]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ISICDataloader:
    def __init__(self, batch_size=128, num_workers=9, img_resize=224, root_dir=None):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_resize = img_resize
        self.root_dir = root_dir

    def run(self, mode, dataset=None):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if mode != "test":
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.img_resize, scale=(0.2, 1.0)),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.8
                    ),
                    transforms.RandomRotation(45),
                    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((self.img_resize, self.img_resize)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

        if dataset:
            all_dataset = dataset
        else:
            all_dataset = ISICDataset(
                root_path=self.root_dir,
                transform=transform,
                mode=mode
            )

        loader = DataLoader(
            dataset=all_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return loader, all_dataset
