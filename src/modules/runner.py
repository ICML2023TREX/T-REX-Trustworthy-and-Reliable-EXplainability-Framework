import torch.nn.functional as F
from torchmetrics import MetricCollection
from tqdm import tqdm
from src.metrics.metrics import *
from src.explanators.deit import VITAttentionGradRollout
import cv2
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
import json
import torch.nn as nn

class BCEWLossConverted:
    def __call__(self, output, target):

        loss = nn.BCELoss()(output,target.to(torch.float32))
        return loss

class SaveRunner:
    def __init__(self, explanator, dataset,metrics):
        self.explanator = explanator
        self.dataset = dataset
        if isinstance(metrics, list) or isinstance(metrics, tuple):
            self.metrics = MetricCollection(*metrics)
        elif isinstance(metrics, MetricCollection):
            self.metrics = metrics
        
    def run(self,model,epoch,save_directory):
        self.labels_used = [0] * 20
        self.metrics.reset()
        criterion = BCEWLossConverted()
        for q,data in enumerate(tqdm(self.dataset)):
            label =data['label']
            if self.labels_used[label] == 40:
                continue
            

            
            self.labels_used[label]+=1
            x = data['image'].unsqueeze(0)
            targets = [ClassifierOutputTarget(label)] * 1

            cam = self.explanator(input_tensor=x, targets=targets)
            grayscale_cam = cam[0]
            label_name = self.dataset.CLASS_LABELS_LIST[label]
            
            full_path=os.path.join(save_directory,self.explanator.__class__.__name__,label_name,str(self.labels_used[label]))
           
            if not os.path.exists(full_path):
                os.makedirs((full_path), exist_ok=True)
            if not os.path.exists(os.path.join(full_path,'correct.jpg')):
                cv2.imwrite(os.path.join(full_path,'correct.jpg'),255*data['image'].numpy().swapaxes(0,1).swapaxes(1,2)[:,:,::-1])

            if not os.path.exists(os.path.join(full_path,'gt.jpg')):
                try:
                    cv2.imwrite(os.path.join(full_path,'gt.jpg'),255*data['attn'].numpy().swapaxes(0,1).swapaxes(1,2)[:,:,::-1])
                except:
                    cv2.imwrite(os.path.join(full_path,'gt.jpg'),255*data['attn'].numpy())

            attn_map_pred = torch.tensor(cam[np.newaxis,:,:,:])
            attn_map_true = data['attn']
            attn_map_pred-=attn_map_pred.min()
            attn_map_pred/=attn_map_pred.max()
            acc = self.metrics(attn_map_pred, attn_map_true)
            acc['Average'] = attn_map_pred.mean()
            res = torch.zeros((1,len(self.dataset.CLASS_LABELS_LIST)))
            res[0,label] = 1
            acc['Loss'] = criterion(model(x.cuda()),res.cuda())
            attn_map = None
            attn_map = cv2.normalize(grayscale_cam, attn_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            attn_map = cv2.applyColorMap(attn_map   , cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(full_path,f'{str(epoch)}.jpg'),attn_map)         
            
            
            try:
                data = json.load(open(os.path.join(full_path,'metrics.json')))
                for i in data:
                    data[i].append(acc[i].item())
            except:
                data = {i:[acc[i].item()] for i in acc}
            with open(os.path.join(full_path,'metrics.json'),'w') as f:
                json.dump(data,f)

            

class EvalRunner:
    def __init__(self, explanator, dataset, metrics, device='cpu'):
        self.device = device
        self.explanator = explanator
        self.dataset = dataset
        # Assign metrics
        if isinstance(metrics, list) or isinstance(metrics, tuple):
            self.metrics = MetricCollection(*metrics)
        elif isinstance(metrics, MetricCollection):
            self.metrics = metrics
        self.metrics.to(self.device)
        self.length =  0
    
    def run(self):
        self.length =  0
        self.labels_used = [0] * 20
        self.metrics.reset()
        for q,data in enumerate(tqdm(self.dataset)):
            label =data['label']
            if self.labels_used[label] == 100:
                continue
            self.labels_used[label]+=1
            x = data['image'].unsqueeze(0).to(self.device)
            targets = [ClassifierOutputTarget(data['label'])]
            attn_map_true = data['attn'].to(self.device)
            attn_map_pred = torch.tensor(self.explanator(input_tensor=x, targets=targets)[np.newaxis,:,:,:]).to(self.device)
            attn_map_pred-=attn_map_pred.min()
            attn_map_pred/=attn_map_pred.max()
            acc = self.metrics(attn_map_pred, attn_map_true)
            self.length+=1

        data = self.metrics.compute()
        self.metrics.reset()
        return data
    def save_metrics(self,metrics,to_save):
        import json
        with open(to_save, 'w') as fp:
            try:
                json.dump({metric_name:metrics[metric_name].item()/self.length for metric_name in metrics}, fp)
            except:
                json.dump({metric_name:metrics[metric_name] for metric_name in metrics}, fp)


class LabelRunner:
    def __init__(self, explanator, dataset,cam_metric):
        self.explanator = explanator
        self.dataset = dataset
        self.cam_metric = cam_metric
        
    def run(self,model):
        self.scores_each_class = {i:[] for i in self.dataset.CLASS_LABELS_LIST}
        self.labels_used = [0] * 20
     
        for q,data in enumerate(tqdm(self.dataset)):
            label =data['label']
            if self.labels_used[label] == 100:
                continue

            self.labels_used[label]+=1
            x = data['image'].unsqueeze(0).cuda()
            targets = [ClassifierOutputTarget(label)] * 1
            grayscale_cams = self.explanator(input_tensor=x, targets=targets)

            scores = self.cam_metric(x, grayscale_cams, targets, model)
            score = scores[0].item()
            self.scores_each_class[self.dataset.CLASS_LABELS_LIST[label]].append(score)
    
        return self.scores_each_class