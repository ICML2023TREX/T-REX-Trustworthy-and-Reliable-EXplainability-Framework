import argparse
import os
import cv2
import torch.nn.functional as F
from torchmetrics import MetricCollection
from tqdm import tqdm
import numpy as np
from src.metrics.metrics import *
from src.models.resnet import resnet50_builder
from src.modules.classification import ClassificationModule
from src.utils.get_config import get_config,get_config_original
import datetime
from src.utils.get_model import get_model
from src.explanators.vit import VitAttention
from captum.attr import LayerGradCam
from src.modules.runner import EvalRunner


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", help="Config File")
    parser.add_argument("--model", help="Model")
    config, dataset = get_config_original(parser.parse_args())
    model_name = get_model(parser.parse_args(),num_classes=config.num_classes,last_layer=config.last_layer,final_activation=config.final_activation)
    model = ClassificationModule(config, model_name)
    model.eval().to(config.device)
    
    
    #explanator = VitAttention(model)
    explanator = LayerGradCam(model,model.model.model[0].features[16])

    metrics = config.metrics



    runner = EvalRunner(explanator, dataset, metrics, config.device)
    print("HERE")
    metrics = runner.run()
    runner.save_metrics(metrics,os.path.join(config.log_dir,f'metrics_{datetime.datetime.now()}.json'))