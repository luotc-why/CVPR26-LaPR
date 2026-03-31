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
def soft_topk(p_soft, topk=3, temperature=1, power=2, beta=1e-2, eps=1e-12):
    K = p_soft.size(-1)
    k = min(topk, K)
    scores = p_soft.clamp_min(eps).pow(power)
    _, idx = torch.topk(scores, k=k, dim=-1)
    mask = torch.zeros_like(scores).scatter_(-1, idx, 1.0)
    logits = scores.add(eps).log() + (1.0 - mask) * math.log(max(beta, eps))
    logits = logits / max(temperature, eps)
    p_used = torch.softmax(logits, dim=-1)
    return p_used, idx


class Router(nn.Module):
    def __init__(self, in_dim, k, hidden=256,
                 topk=3, temperature=0.1, power=4, beta=1e-2, eps=1e-12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, k)
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.topk = topk
        self.temperature = temperature
        self.power = power
        self.beta = beta
        self.eps = eps

    def forward(self, feat_backbone):   # (B, in_dim)
        logits = self.net(feat_backbone)         # (B, K)
        p_soft = F.softmax(logits, dim=-1)       # (B, K)
        p_used, idx = soft_topk(
            p_soft,
            topk=self.topk,
            temperature=self.temperature,
            power=self.power,
            beta=self.beta,
            eps=self.eps,
        )                                        # (B, K), (B, topk)
        return p_used, idx, p_soft              


def build_head(in_dim, out_dim, head_type="mlp"):
    if head_type == "linear":
        return nn.Linear(in_dim, out_dim)
    elif head_type == "mlp":
        return nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
        )
    else:
        raise NotImplementedError(f"head not supported: {head_type}")


class SupVitMLPMoE(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='vit_large_patch14_clip_224', data_base_path = '/data1/liyusheng/CVPR-LRKL/Data', feat_dim=512, K=10, head_type='mlp', router_hidden=256, router_topk=3, router_temperature=1, router_power=2, router_beta=1e-2):
        super(SupVitMLPMoE, self).__init__()
        pretrained_cfg = timm.models.create_model('vit_large_patch14_clip_224').default_cfg
        pretrained_cfg['file'] = f'{data_base_path}/weights/visual_prompt_retrieval/tools/vit_large_patch14_clip_224.laion2b/open_clip_pytorch_model.bin'
        self.encoder = timm.create_model(name, pretrained=True,pretrained_cfg=pretrained_cfg, num_classes=0)
        self.encoder.reset_classifier(num_classes = 0)
        self.encoder.requires_grad_(False)
        self.in_dim = self.encoder.num_features  
        self.feat_dim = feat_dim
        self.K = K
        self.router = Router(
            self.in_dim, K, hidden=router_hidden,
            topk=router_topk, temperature=router_temperature,
            power=router_power, beta=router_beta
        )

        self.heads = nn.ModuleList([build_head(self.in_dim, feat_dim, head_type=head_type) for _ in range(K)])

    def freeze_stages(self):
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        
    def apply_heads_stack(self, feat):  
        outs = [h(feat) for h in self.heads]                  # list of [B, d]
        outs = torch.stack(outs, dim=1)                       # [B, K, d]
        return F.normalize(outs, dim=-1)            
    
    def forward_p(self, x_p, y_p):
        feat_xp_backbone = self.encoder(x_p)
        feat_yp_backbone = self.encoder(y_p)
        feat_p_backbone = feat_xp_backbone + feat_yp_backbone
        feats_p_k = []
        for k in range(self.K):
            fp = self.heads[k](feat_p_backbone)   # (B, d)
            fp = F.normalize(fp, dim=-1)
            feats_p_k.append(fp)
        feats_p_k = torch.stack(feats_p_k, dim=1)     # (B, K, d)
        return feats_p_k
    
    def forward_q(self, x_q):
        feat_q_backbone = self.encoder(x_q)
        gating_weights, idx_q, _  = self.router(feat_q_backbone)  # (B, K)
        feats_q_k = []
        for k in range(self.K):
            fq = self.heads[k](feat_q_backbone)   # (B, d)
            fq = F.normalize(fq, dim=-1)
            feats_q_k.append(fq)
        feats_q_k = torch.stack(feats_q_k, dim=1)     # (B, K, d)
        feat_q = (feats_q_k * gating_weights.unsqueeze(-1)).sum(dim=1)   # (B, d)
        feat_q = F.normalize(feat_q, dim=-1)
        return feat_q, gating_weights
    
    def forward(self, x_q, x_p, y_p):
        feat_q_backbone = self.encoder(x_q)
        feat_xp_backbone = self.encoder(x_p)
        feat_yp_backbone = self.encoder(y_p)
        feat_p_backbone = feat_xp_backbone + feat_yp_backbone
        gating_weights, idx_q, _  = self.router(feat_q_backbone)  # (B, K)
        feats_p_k = []
        for k in range(self.K):
            fp = self.heads[k](feat_p_backbone)   # (B, d)
            fp = F.normalize(fp, dim=-1)
            feats_p_k.append(fp)
        feats_p_k = torch.stack(feats_p_k, dim=1)     # (B, K, d)
        feat_p = (feats_p_k * gating_weights.unsqueeze(-1)).sum(dim=1)   # (B, d)
        feat_p = F.normalize(feat_p, dim=-1)

        feats_q_k = []
        for k in range(self.K):
            fq = self.heads[k](feat_q_backbone)   # (B, d)
            fq = F.normalize(fq, dim=-1)
            feats_q_k.append(fq)
        feats_q_k = torch.stack(feats_q_k, dim=1)     # (B, K, d)
        feat_q = (feats_q_k * gating_weights.unsqueeze(-1)).sum(dim=1)   # (B, d)
        feat_q = F.normalize(feat_q, dim=-1)
        
        return feat_q, feat_p, gating_weights
    

    
    def constrative_forward(self, x_q, x_p1, y_p1, x_p2, y_p2):
        
        X = torch.cat([x_q, x_p1, x_p2, y_p1, y_p2], dim=0)
        with torch.no_grad():                     
            F_all = self.encoder(X)                                
        feat_q_backbone, feat_xp1_backbone, feat_xp2_backbone, feat_yp1_backbone, feat_yp2_backbone = torch.tensor_split(F_all, 5, dim=0)

        feat_p1_backbone = feat_xp1_backbone + feat_yp1_backbone
        feat_p2_backbone = feat_xp2_backbone + feat_yp2_backbone
        gating_weights, idx_q, _  = self.router(feat_q_backbone)  # (B, K)
        feats_p = torch.cat([feat_p1_backbone, feat_p2_backbone, feat_q_backbone], dim=0)  # [3B, D]
        feats_all_k = self.apply_heads_stack(feats_p)                  # [3B, K, d]
        feats_p1_k, feats_p2_k, feats_q_k = torch.tensor_split(feats_all_k, 3, dim=0)      # [B, K, d]
        feat_p1 = (feats_p1_k * gating_weights.unsqueeze(-1)).sum(dim=1)   # (B, d)
        feat_p1 = F.normalize(feat_p1, dim=-1)
        feat_p2 = (feats_p2_k * gating_weights.unsqueeze(-1)).sum(dim=1)   # (B, d)
        feat_p2 = F.normalize(feat_p2, dim=-1)
        feat_q = (feats_q_k * gating_weights.unsqueeze(-1)).sum(dim=1)   # (B, d)
        feat_q = F.normalize(feat_q, dim=-1)
        
        return feat_q, feat_p1, feat_p2, gating_weights
        
def set_train_parts_moe(model: SupVitMLPMoE, train_mlp: bool, train_router: bool):
    for p in model.router.parameters():
        p.requires_grad = train_router

    for m in model.heads:
        for p in m.parameters():
            p.requires_grad = train_mlp
