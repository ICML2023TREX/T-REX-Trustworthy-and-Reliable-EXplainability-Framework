import glob
import json
import os
from os.path import join as pjoin
from typing import Optional, Tuple, Union

import kornia
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import read_image
from .dataset import ExplainabilityDataModule
import matplotlib.pyplot as plt

class SceneClassificationDataModule(ExplainabilityDataModule):
    def __init__(self, root_path: str,
                 train_resize_size: Optional[Union[int, Tuple[int, int]]] = None,
                 eval_resize_size: Optional[Union[int, Tuple[int, int]]] = None,
                 num_workers: int = -1, batch_size: int = 32):
        super(SceneClassificationDataModule, self).__init__(root_path,SceneClassificationDataset,train_resize_size,eval_resize_size,num_workers,batch_size)







class SceneClassificationDataset(Dataset):
    CLASS_LABELS = {
        'nature': 0,
        'urban': 1
    }

    def __init__(self, root_path: str, split: str = 'train',
                 resize_size: Optional[Union[int, Tuple[int, int]]] = None):
        self.root_path = root_path
        self.split = split
        self.resize_size = resize_size
        if isinstance(self.resize_size, int):
            self.resize_size = (self.resize_size, self.resize_size)

        self.image_list = glob.glob(pjoin(root_path, split, '*', '*.jpg'))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image = read_image(image_path)

        image_t = kornia.image_to_tensor(image, keepdim=True).float() / 255.
        if self.resize_size is not None:
            image_t = kornia.geometry.resize(image_t, size=self.resize_size)

        # parse label
        label = image_path.split(os.path.sep)[-2]
        label = SceneClassificationDataset.CLASS_LABELS[label.split('_')[0]]

        return image_t, torch.tensor(label).to(torch.long)



class SceneClassificationAttentionDataset(Dataset):
    CLASS_LABELS = {
        'nature': 0,
        'urban': 1
    }
    CLASS_LABELS_LIST = ['nature','urban']
    ATTN_IGNORE_VALUE = -1

    def __init__(self, root_path: str, split: str = 'train',
                 resize_size: Optional[Union[int, Tuple[int, int]]] = None):
        self.root_path = root_path
        self.split = split
        self.resize_size = resize_size
        if isinstance(self.resize_size, int):
            self.resize_size = (self.resize_size, self.resize_size)

        # parse attention maps into data frame
        image_list = glob.glob(pjoin(root_path, split, '*', '*.jpg'))

        df = pd.Series(image_list, name='path')
        df = pd.DataFrame(df)
        df['id'] = df['path'].str.split('/').str[-1].str[:-4]
        df['label'] = df['path'].str.split('/').str[-2].str.split('_').str[0]
        df['factual_attn'] = None
        df['counterfactual_attn'] = None
        df.set_index('id', inplace=True)

        for attn_path in glob.glob(pjoin(root_path, 'attention_label', '*.csv')):
            is_counterfactual = 'counterfactual' in attn_path
            df_attn = pd.read_csv(attn_path, index_col=0)
            for _, row in df_attn.iterrows():
                if row.img_check == 'good' and row.img_idx in df.index:
                    if is_counterfactual:
                        df.loc[row.img_idx, 'counterfactual_attn'] = row.attention
                    else:
                        df.loc[row.img_idx, 'factual_attn'] = row.attention

        # filter cases with no explanations
        self.df = df[~(df['counterfactual_attn'].isna() & df['factual_attn'].isna())]

    def __len__(self):
        return self.df.shape[0]

    @staticmethod
    def parse_attn_map(attn_map_str: str, image_size: Tuple[int, int]):
        attn_map = np.array(json.loads(attn_map_str))
        attn_map_t = kornia.image_to_tensor(attn_map, keepdim=True).float()
        attn_map_t = kornia.geometry.resize(attn_map_t, size=image_size, interpolation='nearest')
        return attn_map_t.squeeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['path']
        image = read_image(image_path)

        image_t = kornia.image_to_tensor(image, keepdim=True).float() / 255.
        if self.resize_size is not None:
            image_t = kornia.geometry.resize(image_t, size=self.resize_size)

        pos_attn = self.parse_attn_map(row['factual_attn'], (image_t.size(1), image_t.size(2))) \
            if row['factual_attn'] is not None else None
     
        attn = torch.full((image_t.size(-2), image_t.size(-1)), fill_value=0)

        if pos_attn is not None:
            attn[pos_attn.type(torch.bool)] = 1


        label = SceneClassificationAttentionDataset.CLASS_LABELS[row['label']]

        return {
            'image': image_t,
            'attn': attn,  # 0 negative, 1 - positive
            'label': label
        }

