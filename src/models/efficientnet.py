import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s,EfficientNet_V2_S_Weights
from typing import Optional, Any, Callable, List, Type, Union
from torch import Tensor
def efficientnet_builder(num_classes: int = 1000,last_layer=True):
    model = efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.DEFAULT,
        num_classes=1000
    )

    # replace last classification layer
    if last_layer:
        model = torch.nn.Sequential(model, torch.nn.Linear(1000,num_classes))
    
        return model
    else:
        model.classifier[1] = nn.Linear(in_features=1280,out_features=num_classes,bias=True)
        model = torch.nn.Sequential(model) 

        return model


class Efficientnet(nn.Module):
    def __init__(self,num_classes=1000,
                    final_activation: Union[nn.Sigmoid(),nn.Softmax()] = nn.Sigmoid(),last_layer=True):
        super().__init__()
        self.model =  efficientnet_builder(num_classes,last_layer=last_layer)
        self.final_activation = final_activation
        #self.last_conv_layer = self.model[0].features[8]

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        x = self.final_activation(x)
        return x        

