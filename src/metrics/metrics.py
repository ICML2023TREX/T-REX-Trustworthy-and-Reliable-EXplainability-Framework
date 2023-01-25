"""Explanation maps evaluation metrics"""
import torch
from torchmetrics import Metric
import torch.nn.functional as F
class MyAccuracy(Metric):
    '''
    Torchmetric for computing IoU score for our dataset
    '''

    def __init__(self,attn_threshold=0.5):
        super().__init__()
        self.add_state("Accuracy", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("number_of_elements", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.attn_threshold = attn_threshold
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds > self.attn_threshold
        if len(target.shape) == 1:
            target = F.one_hot(target,num_classes=2)
        for i in range(preds.shape[0]):
            intersection = ((preds[i] == 1) & target[i]).sum().item()
            union = ((target[i] == 1) | preds[i]).sum().item()
            self.Accuracy+=intersection/union
            self.number_of_elements+=1
    def compute(self):
        return self.Accuracy/self.number_of_elements


class IoUPreprocessing:
    def __init__(self,attn_threshold=0.5):
        self.attn_threshold = attn_threshold

    def __call__(self,preds,target):
        preds = (preds > self.attn_threshold).squeeze(0).squeeze(0)
        target = (target > self.attn_threshold).squeeze(0).squeeze(0)
        return preds,target

    def __eq__(self, other):
        return self.attn_threshold == other.attn_threshold

    def __hash__(self):
        return 1


class MetricIoU(Metric):
    '''
    Torchmetric for computing IoU score for our dataset
    '''

    def __init__(self,attn_threshold=0.5):
        super().__init__()
        self.preprocessing = IoUPreprocessing(attn_threshold=attn_threshold)
        self.add_state("IoU", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        In future this function could be redesigned to avoid additional computation
        """
        preds,target = self.preprocessing(preds,target)
        intersection = ((preds == 1) & target).sum().item()
        union = ((target == 1) | preds).sum().item()
        self.IoU += intersection / (union + 1e-6)
        assert intersection / (union + 1e-6) <= 1.1

    def compute(self):
        return self.IoU


class MetricF1Score(Metric):
    '''
    Torchmetric for computing IoU score for our dataset
    '''

    def __init__(self,attn_threshold=0.5):
        super().__init__()
        self.preprocessing = IoUPreprocessing(attn_threshold=attn_threshold)
        self.add_state("F1score", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        In future this function could be redesigned to avoid additional computation
        """
        preds, target = self.preprocessing(preds, target)
        tp = ((target == 1) & preds).sum().item()
        fp = ((target == 0) & preds).sum().item()
        fn = ((target == 1) & ~preds).sum().item()
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        self.F1score += 2 * precision * recall / (precision + recall + 1e-6)
        assert 2 * precision * recall / (precision + recall + 1e-6) <= 1.1
    def compute(self):
        return self.F1score


class MetricPrecision(Metric):
    '''
    Torchmetric for computing IoU score for our dataset
    '''

    def __init__(self,attn_threshold=0.5):
        super().__init__()
        self.preprocessing = IoUPreprocessing(attn_threshold=attn_threshold)
        self.add_state("precision", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        In future this function could be redesigned to avoid additional computation
        """
        preds, target = self.preprocessing(preds, target)
        tp = ((target == 1) & preds).sum().item()
        fp = ((target == 0) & preds).sum().item()
        self.precision += tp / (tp + fp + 1e-6)
        assert tp / (tp + fp + 1e-6)  <= 1.1
    def compute(self):
        return self.precision


class MetricRecall(Metric):
    '''
    Torchmetric for computing IoU score for our dataset
    '''

    def __init__(self,attn_threshold=0.5):
        super().__init__()
        self.preprocessing = IoUPreprocessing(attn_threshold=attn_threshold)
        self.add_state("recall", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        In future this function could be redesigned to avoid additional computation
        """
        preds, target = self.preprocessing(preds, target)
        tp = ((target == 1) & preds).sum().item()
        fn = ((target == 1) & ~preds).sum().item()
        recall = tp / (tp + fn + 1e-6)
        self.recall += recall
        assert recall <= 1.1

    def compute(self):
        return self.recall


class MetricMAE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("mae", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        In future this function could be redesigned to avoid additional computation
        """
        preds = torch.nan_to_num(preds,0)
        self.mae += torch.abs(target - preds).sum().item() / \
                    (torch.ones(target.size()).sum().item() + 1e-8)

    def compute(self):
        return self.mae


class MetricMAEFP(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("maeFP", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        In future this function could be redesigned to avoid additional computation
        """

        preds = torch.nan_to_num(preds,0)
        self.maeFP += torch.where(target > 0, 0, preds).sum().item() / \
                      (torch.where(target > 0, 0, 1).sum().item() + 1e-6)

    def compute(self):
        return self.maeFP


class MetricMAEFN(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("maeFN", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        In future this function could be redesigned to avoid additional computation
        """
        preds = torch.nan_to_num(preds,0)
       
        self.maeFN += torch.abs(torch.where(target > 0, preds, 0) - torch.where(target > 0, target, 0)).sum().item() / \
                      (torch.where(target > 0, 1, 0).sum().item() + 1e-6)
    def compute(self):
        return self.maeFN

