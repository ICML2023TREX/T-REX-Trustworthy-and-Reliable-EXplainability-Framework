import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
class ExplainabilityModule(pl.LightningModule):
    def __init__(self, model, criterion, accuracy, augumentation=None, normalization=None):
        super(ExplainabilityModule, self).__init__()
        self.model = model
        self.criterion = criterion
        self.accuracy = accuracy
        self.train_metrics = self.accuracy.clone(prefix='train_')
        self.valid_metrics = self.accuracy.clone(prefix='val_')
        self.test_metrics = self.accuracy.clone(prefix='test_')
        self.valid_loss = 0
        self.test_loss = 0
        self.augmentation = augumentation
        self.normalization = normalization

    def _step(self, batch, augment=False):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._step(batch, augment=True)
        output = self.train_metrics(y_pred, y_true)
        #output = self.train_metrics(nn.Sigmoid()(y_pred), y_true)
        output['train_loss'] = loss
        self.log_dict(output, prog_bar=True, sync_dist=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._step(batch)
        self.valid_metrics.update(y_pred, y_true)
        #self.valid_metrics.update(nn.Sigmoid()(y_pred),y_true)
        self.valid_loss += loss.item() * batch[-1].shape[0]

        return loss

    def test_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._step(batch)
        self.test_metrics.update(y_pred, y_true)
        #self.test_metrics.update(nn.Sigmoid()(y_pred),y_true)

        self.test_loss += loss.item() * batch[-1].shape[0]
        return loss

    def validation_epoch_end(self, outputs):
        output = self.valid_metrics.compute()
        output['val_loss'] = self.valid_loss
        self.valid_loss = 0
        self.log_dict(output, prog_bar=True, sync_dist=True, on_epoch=True)

    def test_epoch_end(self, outputs):
        output = self.test_metrics.compute()
        output['test_loss'] = self.test_loss
        self.log_dict(output, prog_bar=True, sync_dist=True, on_epoch=True)
