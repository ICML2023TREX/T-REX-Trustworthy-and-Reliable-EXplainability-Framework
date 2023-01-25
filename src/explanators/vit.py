import torch
from src.models.visionl16 import VisionTransformer
import numpy as np

class VitAttention:
    def __init__(self,model: VisionTransformer,last_conv_layer):
        self.model = model
    def attribute(self,x,y,**kwargs):
        all_masks = []
        for data in x:
            with torch.no_grad():
                logits, att_mat = self.model(data.unsqueeze(0),attn_layer=True)

            att_mat = torch.stack(att_mat).squeeze(1)

            att_mat = torch.mean(att_mat, dim=1)
            try:
                device = att_mat.get_device()
                device = 'cuda:0'
            except:
                device = 'cpu'
            residual_att = torch.eye(att_mat.size(1)).to(device)
            aug_att_mat = att_mat + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
            aug_att_mat = aug_att_mat[3:]
            joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
            joint_attentions[0] = aug_att_mat[0]
            for n in range(1, aug_att_mat.size(0)):
                joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])    
            v = joint_attentions[-1]
            grid_size = int(np.sqrt(aug_att_mat.size(-1)))
            mask = v[0, 1:].reshape(grid_size, grid_size)
            mask = mask.detach().cpu().numpy()
            mask = mask[np.newaxis,:,:]
            all_masks = all_masks + [mask]
        all_masks = np.array(all_masks)
        return torch.from_numpy(all_masks).to(device)