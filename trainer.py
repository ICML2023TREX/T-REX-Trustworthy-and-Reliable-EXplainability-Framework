import argparse
from src.modules.classification import ClassificationModule
from src.models.equivariant_WRN import Wide_ResNet
from src.utils.train_utils import get_training_loggers, get_training_callbacks
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from src.models.resnet import resnet50_builder
from src.utils.get_config import get_config,get_config_original
from src.utils.get_model import get_model
import torch
 
import os
from datetime import datetime
ATTN_THRESHOLD = 0.5
if __name__ == '__main__':


    # model
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", help="Config File")
    parser.add_argument("--model", help="Config File")
    config,dataset = get_config_original(parser.parse_args())
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    model_name = get_model(parser.parse_args(),config.num_classes,config.last_layer,final_activation=config.final_activation)
    model = ClassificationModule(config, model_name)
    if os.environ.get('LOCAL_RANK', 0) == 0:
        experiment_name = config.experiment_name
        log_path = os.path.join(config.log_dir, config.task, experiment_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

    else:
        experiment_name, log_path = '', ''

    # explanator
    
    callbacks = get_training_callbacks(config, log_path, experiment_name)
    loggers = get_training_loggers(config, log_path, experiment_name)
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        strategy=DDPPlugin(find_unused_parameters=False),
        gradient_clip_val=config.grad_clip_val,
        num_sanity_val_steps=5,
        callbacks=callbacks,
        logger=loggers,
        accelerator=config.device,
        log_every_n_steps=100,
        gpus=config.gpus
    )

    dm = config.datamodule(
            root_path=config.root_path,
            train_resize_size=config.train_resize_size,
            eval_resize_size=config.eval_resize_size,
            num_workers=config.num_workers,
            batch_size=config.batch_size_per_gpu
    )
    
    trainer.fit(model, dm)

    torch.save(model, os.path.join(log_path,'last_train.ckpt'))    