import torch
import torch.nn as nn
from torchvision.models import vit_b_16,ViT_B_16_Weights
from typing import Optional, Any, Callable, List, Type, Union
from torch import Tensor
def deit_builder(num_classes: int = 1000,last_layer=True):
    model = torch.hub.load('facebookresearch/deit:main', 
    'deit_tiny_patch16_224', pretrained=True)

    # replace last classification layer
    if last_layer:
        model = torch.nn.Sequential(model, torch.nn.Linear(1000,num_classes))

        return model
    else:
        model.head = nn.Linear(in_features=192,out_features=num_classes,bias=True)
        model = torch.nn.Sequential(model) 
        return model

class DEIT(nn.Module):
    def __init__(self,num_classes=1000,
                    final_activation: Union[nn.Sigmoid(),nn.Softmax()] = nn.Sigmoid(),last_layer:bool=True):
        super().__init__()
        self.model =  deit_builder(num_classes,last_layer=last_layer)
        self.final_activation = final_activation
        self.last_conv_layer=None
    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        x = self.final_activation(x)
        return x        


