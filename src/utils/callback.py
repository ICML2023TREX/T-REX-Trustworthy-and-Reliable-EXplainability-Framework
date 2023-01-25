from pytorch_lightning.callbacks import Callback
from src.modules.runner import EvalRunner,SaveRunner,LabelRunner
from src.explanators.deit import VITAttentionGradRollout
import tqdm
import torch
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform
import os
import cv2
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
from pytorch_grad_cam.metrics.road import ROADLeastRelevantFirstAverage, ROADMostRelevantFirstAverage
import json
import torch.nn as nn
from torchmetrics import MetricCollection
class ModelEvaluationCallback(Callback):
    def __init__(self,explanator,dataset_eval,save_file,metrics,run_every_x=10):
        self.explanator = explanator
        
        self.dataset_eval = dataset_eval
        self.save_file = save_file
        if isinstance(metrics, list) or isinstance(metrics, tuple):
            self.metrics = MetricCollection(*metrics)
        elif isinstance(metrics, MetricCollection):
            self.metrics = metrics
        
        self.results = {metric_name.__class__.__name__ : [] for metric_name in metrics}
        self.run_every_x=run_every_x
        self.epoch = 0
    
    def on_train_epoch_start(self, trainer, pl_module):

        self.epoch+=1
        if self.run_every_x != 1 and self.epoch % self.run_every_x !=1:
            return

        model = pl_module
        model.eval()
        target_layers = [model.model.model[0].blocks[-1].norm1]

        cam = self.explanator(model=model, target_layers=target_layers, use_cuda=True,reshape_transform=vit_reshape_transform)
        cam.batch_size=40

     
        runner = EvalRunner(explanator=cam,dataset=self.dataset_eval,metrics=self.metrics,device=model.device)
        metrics_now = runner.run()
        for metric in metrics_now:
            self.results[metric].append(metrics_now[metric].item()/runner.length)
        runner.save_metrics(self.results,to_save=self.save_file)
        model.train()
        self.metrics.reset()



class ModelImageSaveCallback(Callback):
    def __init__(self,explanator,dataset_eval,save_directory,metrics,run_every_x=10):
        self.explanator = explanator
        self.dataset_eval = dataset_eval
        self.save_directory = save_directory
        self.run_every_x = run_every_x
        self.epoch = 0
        if isinstance(metrics, list) or isinstance(metrics, tuple):
            self.metrics = MetricCollection(*metrics)
        elif isinstance(metrics, MetricCollection):
            self.metrics = metrics
        
    def on_train_epoch_start(self, trainer, pl_module):

        self.epoch+=1
        if self.run_every_x != 1 and self.epoch % self.run_every_x !=1:
            return

        model = pl_module
        model.eval()
        target_layers = [model.model.model[0].blocks[-1].norm1]

        cam = self.explanator(model=model, target_layers=target_layers, use_cuda=True,reshape_transform=vit_reshape_transform)
        cam.batch_size=40


        runner = SaveRunner(explanator=cam,dataset=self.dataset_eval,metrics=self.metrics)
        runner.run(model,self.epoch,self.save_directory)
        model.train()
        self.metrics.reset()







class NoLabelCallback(Callback):
    def __init__(self,explanator,dataset_eval,save_file,cam_metric,run_every_x=10):
        self.explanator = explanator
        self.dataset_eval = dataset_eval
        self.save_file = save_file
        self.results = {i:[] for i in dataset_eval.CLASS_LABELS_LIST}
        self.start = True
        self.run_every_x = run_every_x
        self.epoch = 0
        self.cam_metric = cam_metric
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch+=1
        if self.run_every_x != 1 and self.epoch % self.run_every_x !=1:
            return
        model = pl_module
        model.eval()
        target_layers = [model.model.model[0].blocks[-1].norm1]

        cam = self.explanator(model=model, target_layers=target_layers, use_cuda=True,reshape_transform=vit_reshape_transform)
        cam.batch_size=40

        runner = LabelRunner(explanator=cam,dataset=self.dataset_eval,cam_metric= self.cam_metric)
        scores = runner.run(model)
        for i in scores:
            for j in range(len(scores[i])):
                if self.start:
                    self.results[i].append([scores[i][j]])
                else:
                    self.results[i][j].append(scores[i][j])
        self.start=False
        with open(self.save_file, 'w') as fp:
            json.dump(self.results, fp)
    
        model.train()

