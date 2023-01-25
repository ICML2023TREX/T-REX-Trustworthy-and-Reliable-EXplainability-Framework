import glob
import os
from os.path import join as pjoin
from typing import Optional, Tuple, Union

import kornia
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCDetection
from torchvision import transforms
from .utils import read_image
from .dataset import ExplainabilityDataModule

class PvocClassificationDataModule(ExplainabilityDataModule):
    def __init__(self, root_path: str,
                 train_resize_size: Optional[Union[int, Tuple[int, int]]] = None,
                 eval_resize_size: Optional[Union[int, Tuple[int, int]]] = None,
                 num_workers: int = -1, batch_size: int = 32):
        super(PvocClassificationDataModule, self).__init__(root_path,PvocClassificationDataset,train_resize_size,eval_resize_size,num_workers,batch_size)

    def setup(self, stage: Optional[str] = None) -> None:
        self._train_ds = PvocClassificationDataset(self.root_path, image_set='train',
                                                   resize_size=self.train_resize_size)
        self._val_ds = PvocClassificationDataset(self.root_path, image_set='val', resize_size=self.eval_resize_size)



class PvocClassificationDataset(VOCDetection):
    """Multilabel classification wrapper For PascalVOC Detection Dataset"""

    CLASS_LABELS = ['aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self, root, year='2012',
                 image_set='train',
                 download=False,
                 target_transform=None,
                 resize_size: Optional[Union[int, Tuple[int, int]]] = None):
        self.resize_size = resize_size
        if isinstance(self.resize_size, int):
            self.resize_size = (self.resize_size, self.resize_size)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.resize_size),

            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                ])


        super().__init__(
            root,
            year=year,
            image_set=image_set,
            download=download,
            transform=self.transform,
            target_transform=target_transform)

    def __len__(self):
        """
        Returns:
            size of the dataset
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a binary encoded list of classes.
        """
        img, label = super().__getitem__(index)
        # encode labels
        label = {obj['name'] for obj in label['annotation']['object']}

        label = [int(name in label) for name in self.CLASS_LABELS]

        label = torch.Tensor(label).to(torch.int32)
        return img, label











class PvocAttentionDataset(Dataset):
    """Benchmark human attention dataset for PascalVOC labels"""


    CLASS_LABELS_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor']



    def __init__(self, root_path: str,
                 resize_size: Optional[Union[int, Tuple[int, int]]] = None):
        self.root_path = root_path
        self.resize_size = resize_size
        if isinstance(self.resize_size, int):
            self.resize_size = (self.resize_size, self.resize_size)
        maps_subdir = 'Image/PASCAL_VOC_2012/human_attention_mask'
        img_subdir = 'Image/PASCAL_VOC_2012/original_images'
        self.maps = glob.glob(pjoin(root_path, maps_subdir, '*', '*.jpg'))
        self.images = glob.glob(pjoin(root_path, img_subdir, '*', '*.jpg'))
        self.maps = {path.split('/')[-1]: path for path in self.maps}
        self.images = {path.split('/')[-1].replace('cat-', ''): path for path in self.images}
        self.maps = {k: v for k, v in self.maps.items() if k in self.images}


    def __len__(self):
        return len(self.maps)


    def __getitem__(self, idx):
        pair_name = list(self.maps.keys())[idx]

        # get image
        image_path = self.images[pair_name]
        image = read_image(image_path)
        image_t = kornia.image_to_tensor(image, keepdim=True).float() / 255.
        if self.resize_size is not None:
            image_t = kornia.geometry.resize(image_t, size=self.resize_size)

        # get map
        map_path = self.maps[pair_name]
        att_map = read_image(map_path)
        map_t = kornia.image_to_tensor(att_map, keepdim=True).float() / 255.

        if self.resize_size is not None:
            map_t = kornia.geometry.resize(map_t, size=self.resize_size)
        # get label
        label = map_path.split(os.path.sep)[-2]
        label = PvocClassificationDataset.CLASS_LABELS.index(label)

        return {
            'image': image_t,
            'attn': map_t,
            'label': label
        }
