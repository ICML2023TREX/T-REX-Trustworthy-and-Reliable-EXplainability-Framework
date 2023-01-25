import kornia.augmentation as KA
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from src.models.resnet import resnet50_builder
import pickle
from src.modules.module import ExplainabilityModule
from src.models.equivariant_WRN import Wide_ResNet
class ClassificationModule(ExplainabilityModule):
    """
    Multiclass/Multilabel classification training module for ResNet50 module
    with last two pooling layers replaced by dilated convolution
    """
    def __init__(self, config,model):


        self.config = config
        # define loss function
        criterion = config.criterion

        # define accuracy metric
        accuracy = config.accuracy
        augmentation = None
        # define kornia augmentations
        if hasattr(config, 'augmentation'):
            augmentation = config.augmentation

        # imagenet normalization
        normalization = None
        if hasattr(config, 'normalization'):
            normalization = config.normalization
        super(ClassificationModule, self).__init__(model, criterion, accuracy, augmentation, normalization)



    def forward(self, x: torch.Tensor,attn_layer : bool = False) -> torch.Tensor:
        """Run inference on the resnet 50 model"""
        if self.normalization:
            x = self.normalization(x)
        if attn_layer:
            return self.model(x,attn_layer)
        
        return self.model(x)

    def _step(self, batch, augment: bool = False):
        """Generic step for train-/eval-loop"""
        x, y_true = batch
        if augment:
            x = self.augmentation(x)
        y_pred = self(x)
        loss = self.criterion(y_pred, y_true)
        return loss, y_pred,y_true



    def get_parameter_groups(self):
        """
        Return separately backbone and classification head parameters
        """
        backbone = []
        classifier = []
        for name, params in self.model.named_parameters():
            if name.startswith('fc'):
                classifier.append(params)
            else:
                backbone.append(params)
        return backbone, classifier
        
    def configure_optimizers(self):
        try:
            backbone_params, classifier_params = self.get_parameter_groups()
            optimizer = torch.optim.Adam([
                {'params': backbone_params, 'lr': self.config.backbone_lr},
                {'params': classifier_params, 'lr': self.config.classifier_lr}
            ])
        except:
            optimizer = torch.optim.Adam([
                {
                    'params':self.model.model.named_parameters()   , 'lr':self.config.backbone_lr
                }
            ])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=1,
            gamma=self.config.scheduler_gamma
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def last_conv_layer(self):
        return self.model.last_conv_layer