import os

import torch.nn.functional as F
from captum.attr import LayerGradCam
from tqdm import tqdm

from config.resnet_Pvoc_mc import ResNetPvocMC
from src.datasets.Pvoc import PvocAttentionDataset
from src.modules.resnet_classification import ClassificationModule
from src.metric_old import *
from src.metrics.metrics import *
ATTN_THRESHOLD = 0.5
ATTN_THRESHOLD_LABELS = 0.5

device = torch.device('cpu')

# model
config = ResNetPvocMC()
model = ClassificationModule(config)

model.eval().to(device)

# explanator
explanator = LayerGradCam(model, model.model.layer4[2].conv3)

att_ds = PvocAttentionDataset(
    root_path='/home/bovey/Downloads/data/ML-Interpretability-Evaluation-Benchmark')
print(os.listdir())
# evaluate
metric_segmentation = TotalMeter(MetricF1Score, MetricIoU, MetricPrecision, MetricRecall, MetricMAEFP, MetricMAE, MetricMAEFN)

print("HERE")
for data in tqdm(att_ds):
    x = data[0].unsqueeze(0).to(device)
    y = data[2]
    attn_map_true = data[1].to(device)

    # apply relu and normalize to 0..1 range
    attn_map_pred = explanator.attribute(x, y, relu_attributions=True).detach()
    attn_map_pred = F.interpolate(attn_map_pred, size=(attn_map_true.size(-2), attn_map_true.size(-1)))
    attn_map_pred = attn_map_pred / attn_map_pred.max()

    # evaluate with mae metrics
    metric_segmentation.update(attn_map_pred,attn_map_true)
    print(metric_segmentation.compute())
