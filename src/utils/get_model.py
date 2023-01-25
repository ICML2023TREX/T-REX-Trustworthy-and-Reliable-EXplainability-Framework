from src.models.resnet import resnet50_builder
from src.models.equivariant_WRN import Wide_ResNet
from src.models.mobilenetv3 import MobileNet
from src.models.efficientnet import Efficientnet
from src.models.visionl16 import VisionTransformer
from src.models.deit import DEIT
def get_model(args,num_classes,last_layer,final_activation):
    if args.model == 'resnet':
        return resnet50_builder(num_classes=num_classes)
    if args.model == 'widenet':
        return Wide_ResNet(10, 6, 0.3, initial_stride=1, N=8, f=True, r=0, num_classes=num_classes)

    if args.model == 'mobilenet':

        return MobileNet(num_classes = num_classes,last_layer=last_layer,final_activation=final_activation)
    if args.model == 'efficientnet':
        return Efficientnet(num_classes= num_classes,last_layer=last_layer,final_activation=final_activation)
    if args.model == 'vision16':
        return VisionTransformer(num_classes=num_classes,final_activation=final_activation)
    if args.model == 'deit':
        return DEIT(num_classes=num_classes,last_layer=last_layer,final_activation=final_activation)