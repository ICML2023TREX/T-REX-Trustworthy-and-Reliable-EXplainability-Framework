import torch
import torch.nn.functional as F
from captum.attr import LayerGradCam
from tqdm import tqdm

from config.resnet_gender_mc import ResNetGenderMC
from src.datasets.gender import GenderClassificationAttentionDataset
from src.metrics.metrics import TotalMeter, MetricF1Score,MetricIoU,MetricPrecision,MetricRecall
from src.modules.resnet_classification import ClassificationModule

ATTN_THRESHOLD = 0.5

device = torch.device('cpu')

# model
config = ResNetGenderMC()
model = ClassificationModule(config)

model.eval().to(device)

# explanator
explanator = LayerGradCam(model, model.model.layer4[2].conv3)

dataset = GenderClassificationAttentionDataset(
    root_path=config.root_path, split='test', resize_size=config.eval_resize_size)

# evaluate
metric_segmentation = TotalMeter(MetricF1Score,MetricIoU,MetricPrecision,MetricRecall)


for data in tqdm(dataset):
    x = data['image'].unsqueeze(0).to(device)
    y = data['label']
    attn_map_true = data['attn'].to(device)

    # apply relu and normalize to 0..1 range
    attn_map_pred = explanator.attribute(x, y, relu_attributions=True).detach()


    attn_map_pred = F.interpolate(attn_map_pred, size=(attn_map_true.size(-2), attn_map_true.size(-1)))
    attn_map_pred = attn_map_pred / attn_map_pred.max()
    metric_segmentation.update(attn_map_pred,attn_map_true)

    print(metric_segmentation.compute())

