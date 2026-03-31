"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import timm
from  timm.utils import freeze

def build_head(in_dim, out_dim, head_type="mlp"):
    if head_type == "linear":
        return nn.Linear(in_dim, out_dim)
    elif head_type == "mlp":
        return nn.Sequential(
            nn.Linear(in_dim, 2*in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*in_dim, 2*in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*in_dim, out_dim)
        )
    else:
        raise NotImplementedError(f"head not supported: {head_type}")



def add_label_token_forward_features(vit, x, l_x):   
    x = vit.patch_embed(x)
    x = vit._pos_embed(x) # x (B,N,D)   l_x (B,1,D)
    x = torch.cat([x[:, :1], l_x, x[:, 1:]], dim=1)  # add 1 label tokens after cls token
    x = vit.norm_pre(x)
    x = vit.blocks(x)
    x = vit.norm(x)
    return x

def add_label_token_forward(vit, x, l_x):
    x = add_label_token_forward_features(vit,x,l_x)
    x = vit.forward_head(x)
    return x

class SupVitPtuning(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='vit_large_patch14_clip_224', jump = True, data_base_path = '/data1/liyusheng/CVPR-LRKL/Data',
                 feat_dim=128, K=10, head_type='mlp', router_hidden=256, router_topk=3, router_temperature=1, router_power=2, router_beta=1e-2):
        super(SupVitPtuning, self).__init__()

        pretrained_cfg = timm.models.create_model('vit_large_patch14_clip_224').default_cfg
        pretrained_cfg['file'] = f'{data_base_path}/weights/visual_prompt_retrieval/tools/vit_large_patch14_clip_224.laion2b/open_clip_pytorch_model.bin'
        self.encoder = timm.create_model(name, pretrained=True,pretrained_cfg=pretrained_cfg, num_classes=0)
        self.encoder.reset_classifier(num_classes = 0)
        self.encoder.requires_grad_(False)
        self.in_dim = self.encoder.num_features      # more rubust rather than 1024 
        self.feat_dim = feat_dim
        self.jump = jump
        self.MLP_L_I = nn.Linear(self.in_dim, self.in_dim)
        with torch.no_grad():
            self.MLP_L_I.weight.copy_(torch.eye(self.in_dim))
            self.MLP_L_I.bias.zero_()
        self.l_x = nn.Parameter(torch.zeros(1, 1, self.in_dim))   # one label tokens
        self.K = K
        self.head = build_head(self.in_dim, feat_dim, head_type=head_type)
        

    def freeze_stages(self):
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    
    def apply_heads_stack(self, feat):  
        outs = self.head(feat)                   # list of [B, d]
        return F.normalize(outs, dim=-1)            

    def forward_p(self, x_p, y_p):
        yp_emb = self.encoder(y_p) # (B,D)
        yp_emb = self.MLP_L_I(yp_emb)   # (B,D)
        yp_emb = yp_emb.unsqueeze(1)  # (B,1,D)
        feat_p_backbone = add_label_token_forward(self.encoder, x_p, yp_emb) # (B,D)
        if self.jump:
            feat_p_backbone = feat_p_backbone + yp_emb[:,0,:]
        feat_p = self.apply_heads_stack(feat_p_backbone)                  # [B, d]
        return feat_p
    
    def forward_q(self, x_q):
        b, c, h, w = x_q.shape
        l_x_expanded = self.l_x.expand(b, -1, -1)        # (B,1,D)
        feat_q_backbone = add_label_token_forward(self.encoder, x_q, l_x_expanded) # (B,D)
        feat_q = self.apply_heads_stack(feat_q_backbone)                  # [B, d]        
        return feat_q
    
    def constrative_forward(self, x_q, x_p1, y_p1, x_p2, y_p2):
        b, c, h, w = x_q.shape
        l_x_expanded = self.l_x.expand(b, -1, -1)        # (B,1,D)
        feat_q_backbone = add_label_token_forward(self.encoder, x_q, l_x_expanded) # (B,D)
        feat_q = self.apply_heads_stack(feat_q_backbone)                  # [B, d]

        
        Y1 = torch.cat([y_p1, y_p2], dim=0)
        with torch.no_grad():
            F_y1 = self.encoder(Y1) # (2B,D)
        F_y1 = self.MLP_L_I(F_y1)   # (2B,D)
        yp1_emb, yp2_emb= torch.tensor_split(F_y1.unsqueeze(1), 2, dim=0) # (B,1,D)
        
        feat_p1_backbone = add_label_token_forward(self.encoder, x_p1, yp1_emb) # (B,D)
        feat_p2_backbone = add_label_token_forward(self.encoder, x_p2, yp2_emb) # (B,D)

        if self.jump:
            feat_p1_backbone = feat_p1_backbone + yp1_emb[:,0,:]
            feat_p2_backbone = feat_p2_backbone + yp2_emb[:,0,:]
        
        feats_p = torch.cat([feat_p1_backbone, feat_p2_backbone], dim=0)  # [4B, D]
        feats_all_k = self.apply_heads_stack(feats_p)                  # [4B, K, d]
        feat_p1, feat_p2 = torch.tensor_split(feats_all_k, 2, dim=0)      # [B, K, d]
        
        return feat_q, feat_p1, feat_p2

    
        
    def forward(self, x_q, x_p, y_p):
        b, c, h, w = x_q.shape
        l_x_expanded = self.l_x.expand(b, -1, -1)        # (B,1,D)
        feat_q_backbone = add_label_token_forward(self.encoder, x_q, l_x_expanded) # (B,D)
        feat_q = self.apply_heads_stack(feat_q_backbone)                  # [B, d]

        yp_emb = self.encoder(y_p) # (B,D)
        yp_emb = self.MLP_L_I(yp_emb)   # (B,D)
        yp_emb = yp_emb.unsqueeze(1)  # (B,1,D)
        feat_p_backbone = add_label_token_forward(self.encoder, x_p, yp_emb) # (B,D)

        if self.jump: 
            feat_p_backbone = feat_p_backbone + yp_emb[:,0,:]
        
        feat_p = self.apply_heads_stack(feat_p_backbone)                  # [4B, K, d]
        
        return feat_q, feat_p
                    