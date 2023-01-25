import argparse
import os
import torch.nn.functional as F
from captum.attr import LayerGradCam
from torchmetrics import MetricCollection
from tqdm import tqdm
import numpy as np
from src.metrics.metrics import *
from src.models.resnet import resnet50_builder
from src.modules.classification import ClassificationModule
from src.utils.get_config import get_config,get_config_original
import datetime
from src.utils.get_model import get_model
import cv2
from src.explanators.deit import VITAttentionGradRollout

from src.modules.runner import EvalRunnerDeit
if __name__ == '__main__':
    '''
    case = 1
    if case == 0:
        config = ResNetGenderMC()
        dataset = GenderClassificationAttentionDataset(
            root_path=config.root_path, split='test', resize_size=config.eval_resize_size)
    else:
        config = ResNetPvocMC()


        model_name = resnet50_builder(num_classes=config.num_classes)
        model = ResNetClassificationModule(config,model_name)

        model.eval().to(config.device)

        # explanator
        explanator = LayerGradCam(model, model.model.layer4[2].conv3)

        dataset = PvocAttentionDataset(
            root_path='/home/bovey/datasets/data/ML-Interpretability-Evaluation-Benchmark')
'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", help="Config File")
    parser.add_argument("--model", help="Model")
    config, dataset = get_config_original(parser.parse_args())
    model_name = get_model(parser.parse_args(),num_classes=config.num_classes,last_layer=config.last_layer,final_activation=config.final_activation)
    model = ClassificationModule(config, model_name)
    #ckpt = torch.load('/home/bovey/Downloads/gender/logs/deit_best.ckpt', map_location='cpu')
    #state_dict = ckpt['state_dict']
    #print(model.load_state_dict(state_dict, strict=True))
    #model.eval().to(config.device)
    model.eval()
    

    
    metrics = config.metrics
    explanator = VITAttentionGradRollout(model, discard_ratio=0.6, attention_layer_name = 'attn_drop')
    



    runner = EvalRunnerDeit(explanator, dataset, metrics, 'cpu')
    print("HERE")
    metrics = runner.run()
    runner.save_metrics(metrics,os.path.join(config.log_dir,f'metrics_{datetime.datetime.now()}.json'))