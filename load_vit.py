import os
from urllib.request import urlretrieve
if not os.path.isfile("../../vision_model_baseline/ViT-B_16-224.npz"):
    urlretrieve("https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz", "../../vision_model_baseline/ViT-B_16-224.npz")
