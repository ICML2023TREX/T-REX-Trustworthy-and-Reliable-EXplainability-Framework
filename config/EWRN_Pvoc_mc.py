"""Config for equivariant wide resnet module for multilabel Pvoc classification """

from dataclasses import dataclass, field
from typing import List


@dataclass
class EWRNPvocMC:
    """Config for equivariant wide resnet training on PascalVOC multilabel classification task"""
    seed: int = 42
    log_dir: str = '/content/drive/MyDrive/try/logs'
    task: str = 'EWRN_Pvoc_mc'
    gpus: List[int] = field(default_factory=lambda: [0])
    # training
    epochs: int = 50
    grad_clip_val: float = 2.0
    backbone_lr: float = 1e-5
    classifier_lr: float = 1e-4
    scheduler_gamma: float = 0.9997
    # data
    root_path: str = '/content/'
    train_resize_size: int = 256
    train_crop_size: int = 128
    eval_resize_size: int = 128
    num_workers: int = 4
    batch_size_per_gpu: int = 16

    num_classes: int = 20
