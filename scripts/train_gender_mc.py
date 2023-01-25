"""Script for training resnet50 for binary classification"""
import os
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from config.resnet_gender_mc import ResNetGenderMC
from src.datasets.gender import GenderClassificationDataModule
from src.modules.resnet_classification import ClassificationModule
from src.utils.train_utils import get_training_loggers, get_training_callbacks


def main():
    config = ResNetGenderMC()
    pl.seed_everything(int(os.environ.get('LOCAL_RANK', 0)) + config.seed)

    if os.environ.get('LOCAL_RANK', 0) == 0:
        experiment_name = '{}'.format(
            str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        )
        log_path = os.path.join(config.log_dir, config.task, experiment_name)

        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

    else:
        experiment_name, log_path = '', ''

    dm = GenderClassificationDataModule(
        root_path=config.root_path,
        train_resize_size=config.train_resize_size,
        eval_resize_size=config.eval_resize_size,
        num_workers=config.num_workers,
        batch_size=config.batch_size_per_gpu
    )

    model = ClassificationModule(config)

    callbacks = get_training_callbacks(config, log_path, experiment_name)
    loggers = get_training_loggers(config, log_path, experiment_name)

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        strategy=DDPPlugin(find_unused_parameters=False),
        gradient_clip_val=config.grad_clip_val,
        num_sanity_val_steps=5,
        callbacks=callbacks,
        logger=loggers,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
