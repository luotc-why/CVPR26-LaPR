"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


import timm
from  timm.utils import freeze


# from timm.models import VisionTransformer

# model_dict = {
#     'resnet18': 512,
#     'resnet34': 512,
#     'resnet50': 2048,
#     'resnet101': 2048,
# }


class SupVit(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupVit, self).__init__()
        self.encoder = timm.create_model(name, pretrained=True)
        self.encoder.reset_classifier(num_classes = 0)
        #freeze model
        self.encoder.requires_grad_(False)
        dim_in = 1024 #model_dict[name]
        # self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def freeze_stages(self):
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def forward(self, x):
        feat = self.encoder(x)
        # import pdb;pdb.set_trace()
        # feat = self.encoder.module.forward_features(x)
        # feat = self.encoder.module.forward_head(feat,pre_logits=True)
        feat = F.normalize(self.head(feat), dim=1)
        return feat



