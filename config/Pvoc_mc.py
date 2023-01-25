"""Config for resnet module for multilabel Pvoc classification """
import os
from dataclasses import dataclass, field
from typing import List
import torch.nn as nn
from torchmetrics import Accuracy,MetricCollection
import kornia.augmentation as KA
import torch
from src.datasets.Pvoc import PvocClassificationDataModule
from src.metrics.metrics import *
import torch.nn.functional as F
from src.utils.callback import ModelEvaluationCallback,ModelImageSaveCallback,NoLabelCallback

from src.explanators.vit import VitAttention
from src.explanators.deit import VITAttentionGradRollout
from captum.attr import LayerGradCam
from src.datasets.Pvoc import PvocAttentionDataset
from datetime import datetime

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad,LayerCAM
from pytorch_grad_cam.metrics.road import ROADLeastRelevantFirstAverage, ROADMostRelevantFirstAverage,ROADLeastRelevantFirst,ROADMostRelevantFirst
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange

class BCEWLossConverted:
    def __call__(self, output, target):

        loss = nn.BCELoss()(output,target.to(torch.float32))
        return loss

@dataclass
class PvocMC:
    """Config for ResNet-50 training on PascalVOC multilabel classification task"""
    seed: int = 42
    log_dir: str = '/home/User/Downloads/Sigmoid/logs'
    task: str = 'deit'
    device = 'cuda:0'
    gpus = [0]

    # training
    epochs: int = 1000
    grad_clip_val: float = 2.0
    backbone_lr: float = 1e-5
    classifier_lr: float = 1e-5
    scheduler_gamma: float = 0.999
    # data
    root_path: str = '/home/User/datasets/data/data'
    train_resize_size: int = 224
    train_crop_size: int = 224
    eval_resize_size: int = 224

    num_workers: int = 8
    batch_size_per_gpu: int = 64
    num_classes: int = 20
    last_layer = False

    criterion = BCEWLossConverted()
    accuracy = MetricCollection([MyAccuracy(attn_threshold=0.5)])
    augmentation: KA.AugmentationSequential =  KA.AugmentationSequential(
            KA.RandomEqualize(p=0.2),
            KA.RandomSharpness(p=0.2),
            KA.RandomSolarize(p=0.2),
            KA.RandomGaussianNoise(p=0.5, mean=0., std=0.05),
            KA.RandomPerspective(distortion_scale=0.5, p=0.3),
            KA.RandomElasticTransform(p=0.2),
             KA.RandomCrop((train_crop_size, train_crop_size), p=1.0)
        )
    datamodule = PvocClassificationDataModule
    metrics=[MetricF1Score(attn_threshold=0.5), MetricIoU(), MetricPrecision(attn_threshold=0.5), MetricRecall(), MetricMAE(),
               MetricMAEFN(), MetricMAEFP()]
    final_activation=nn.Sigmoid()

    dataset_eval = PvocAttentionDataset(
            root_path='/home/User/datasets/data/ML-Interpretability-Evaluation-Benchmark',resize_size=eval_resize_size)
    experiment_name = '{}'.format(
        str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    )
    #normalization = KA.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    callbacks=[
    #ModelEvaluationCallback(explanator=GradCAMPlusPlus,dataset_eval=dataset_eval,save_file = os.path.join(log_dir,task,experiment_name,f'GradCAMPlusPlus.json'),metrics=metrics,run_every_x=1),
    #ModelEvaluationCallback(explanator=LayerCAM,dataset_eval=dataset_eval,save_file = os.path.join(log_dir,task,experiment_name,f'LayerCAM.json'),metrics=metrics,run_every_x=1),
    ModelEvaluationCallback(explanator=ScoreCAM,dataset_eval=dataset_eval,save_file = os.path.join(log_dir,task,experiment_name,f'ScoreCAM.json'),metrics=metrics,run_every_x=5),

    #NoLabelCallback(explanator=GradCAMPlusPlus,dataset_eval=dataset_eval,save_file = os.path.join(log_dir,task,experiment_name,f'GradCAMPlusPlusNoLabelRoadLeast.json'),cam_metric = ROADLeastRelevantFirst(percentile=90),run_every_x=10),
    #NoLabelCallback(explanator=LayerCAM,dataset_eval=dataset_eval,save_file = os.path.join(log_dir,task,experiment_name,f'LayerCAMNoLabelRoadLeast.json'),cam_metric = ROADLeastRelevantFirst(percentile=90),run_every_x=5),
    #NoLabelCallback(explanator=GradCAMPlusPlus,dataset_eval=dataset_eval,save_file = os.path.join(log_dir,task,experiment_name,f'GradCAMPlusPlusNoLabelRoadMost.json'),cam_metric = ROADMostRelevantFirst(percentile=90),run_every_x=5),
    #NoLabelCallback(explanator=LayerCAM,dataset_eval=dataset_eval,save_file = os.path.join(log_dir,task,experiment_name,f'LayerCAMNoLabelRoadMost.json'),cam_metric = ROADMostRelevantFirst(percentile=90),run_every_x=5),
    #NoLabelCallback(explanator=LayerCAM,dataset_eval=dataset_eval,save_file = os.path.join(log_dir,task,experiment_name,f'LayerCAMNoLabelConfidenceChange.json'),cam_metric = CamMultImageConfidenceChange(),run_every_x=5),
    NoLabelCallback(explanator=ScoreCAM,dataset_eval=dataset_eval,save_file = os.path.join(log_dir,task,experiment_name,f'ScoreCAMNoLabelConfidenceChange.json'),cam_metric = CamMultImageConfidenceChange(),run_every_x=5),
    NoLabelCallback(explanator=ScoreCAM,dataset_eval=dataset_eval,save_file = os.path.join(log_dir,task,experiment_name,f'ScoreCAMNoLabelRoadMost.json'),cam_metric = ROADMostRelevantFirst(percentile=90),run_every_x=5),
    ModelImageSaveCallback(explanator=ScoreCAM,dataset_eval=dataset_eval,save_directory = os.path.join(log_dir,task,experiment_name,f'photos'),metrics=metrics,run_every_x=5),   
    #ModelImageSaveCallback(explanator=LayerCAM,dataset_eval=dataset_eval,save_directory = os.path.join(log_dir,task,experiment_name,f'photos'),metrics=metrics,run_every_x=1),
    #ModelImageSaveCallback(explanator=GradCAMPlusPlus,dataset_eval=dataset_eval,save_directory = os.path.join(log_dir,task,experiment_name,f'photos'),metrics=metrics,run_every_x=1),
   

    ]
    if not os.path.exists(os.path.join(log_dir,task,experiment_name,f'photos')):
        os.makedirs(os.path.join(log_dir,task,experiment_name,f'photos'), exist_ok=True)
