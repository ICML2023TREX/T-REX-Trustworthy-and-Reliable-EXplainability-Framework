import kornia.augmentation as KA
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from src.models.equivariant_WRN import Wide_ResNet


class EWRNClassificationModule(pl.LightningModule):
    """
    Multiclass/Multilabel classification training module for equivariant wide resnet module
    """

    def __init__(self, config):
        super(EWRNClassificationModule, self).__init__()

        self.config = config
        self.model = Wide_ResNet(10, 6, 0.3, initial_stride=1, N=8, f=True, r=0, num_classes=config.num_classes)

        # define loss function
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')

        # define accuracy metric
        self.accuracy = Accuracy(average='macro', num_classes=config.num_classes)
        self.micro = Accuracy(average='micro', num_classes=config.num_classes)
        self.weighted = Accuracy(average='weighted', num_classes=config.num_classes)
        self.samples = Accuracy(average='samples', num_classes=config.num_classes)
        self.classwise_acc = Accuracy(average='none', num_classes=config.num_classes)

        # define kornia augmentations
        self.augmentation = KA.AugmentationSequential(
            KA.RandomEqualize(p=0.2),
            KA.RandomSharpness(p=0.2),
            KA.RandomSolarize(p=0.2),
            KA.RandomGaussianNoise(p=0.5, mean=0., std=0.05),
            KA.RandomPerspective(distortion_scale=0.5, p=0.3),
            KA.RandomElasticTransform(p=0.2),
            KA.RandomCrop((config.train_crop_size, config.train_crop_size), p=1.0)
        )

        # imagenet normalization
        self.normalization = KA.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_parameter_groups(self):
        """
        Return separately backbone and classification head parameters
        """
        backbone = []
        classifier = []
        for name, params in self.model.named_parameters():
            if name.startswith('linear'):
                classifier.append(params)
            else:
                backbone.append(params)
        return backbone, classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference on the equivariant wide resnet model"""
        x = self.normalization(x)
        return self.model(x)



    def configure_optimizers(self):
        backbone_params, classifier_params = self.get_parameter_groups()
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': self.config.backbone_lr},
            {'params': classifier_params, 'lr': self.config.classifier_lr}
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
                'interval': 'step',
                'frequency': 1
            }
        }
