"""
Extract features for SupPR.
"""
import os
import sys
import glob
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
import torchvision.models as models
from torchvision import transforms as T
from torch.nn import functional as F

import timm
from timm.models import load_checkpoint
from collections import OrderedDict

from VitMoE_XY import SupVitMLPMoE

pretrain_name = sys.argv[1]
feature_name = sys.argv[2]
split = sys.argv[3]
data_base_path = sys.argv[4]
lr = sys.argv[5]
checkpoint_epoch = sys.argv[6]

def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.','') if 'module.' in k else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict
     
# load the image transformer
t = []
size = 224
t.append(T.Resize((size,size), interpolation=Image.BICUBIC))
t.append(T.CenterCrop(size))
t.append(T.ToTensor())
t.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
center_crop = T.Compose(t)

save_dir = f"{data_base_path}/figures_dataset/imagenet/{feature_name}_col_lr_{lr}_epoch_{checkpoint_epoch}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    print(f"Directory exists at {save_dir}")
    # sys.exit()
# assert False

# meta_root =  f"/data1/liyusheng/CVPR-LRKL/Codes/evaluate/splits/pascal/{split}"

if split == 'val':
    image_root = f"{data_base_path}/figures_dataset/imagenet/test_data"
    ann_root = f"{data_base_path}/figures_dataset/imagenet/test_label"
else :
    image_root = f"{data_base_path}/figures_dataset/imagenet/train_data"
    ann_root = f"{data_base_path}/figures_dataset/imagenet/train_label"

model = SupVitMLPMoE(pretrain_name,data_base_path=data_base_path)
ckpt_path = f"{data_base_path}/save/col_SupCon_moe_moe/path_moe_moe_models/SupCon_path_{pretrain_name}_folder_0_seed_0_lr_{lr}_decay_0.0001_cropsz_{size}_bsz_64_temp_0.1_trial_0_cosine_pretrain-vit-freeze-encoder/{checkpoint_epoch}.pth"
try:
    model.load_state_dict(clean_state_dict(torch.load(ckpt_path)['model']))
except:
    print('{} is wrong'.format(ckpt_path))
    sys.exit()
model.eval()
model = model.cuda()
examples = sorted([os.path.join(image_root, file) for file in os.listdir(image_root)])
if len(examples) == 0:
    print(f"zeros folder")
    sys.exit()

if split == 'trn':
    image_names = [example.strip()[:-4] for example in examples]
    ann_examples = [os.path.join(ann_root, example.split('/')[-1]) for example in examples]
    examples = [os.path.join(image_root, example) for example in examples]
    
if split == 'val':
    image_names = [example.strip()[:-5] for example in examples]
    ann_examples = [os.path.join(ann_root, example.split('/')[-1]) for example in examples]
    examples = [os.path.join(image_root, example) for example in examples]

imgs = []

if split == 'trn':
    labels = []

global_features = torch.tensor([]).cuda()
global_gatings = torch.tensor([]).cuda()

for example, ann_example, image_name in tqdm(zip(examples,ann_examples,image_names)):
    try:
        path = os.path.join(example)
        img = Image.open(path).convert("RGB")
        img = center_crop(img)
        imgs.append(img)
        if split == 'trn':
            label = Image.open(ann_example).convert('RGB')
            label = center_crop(label)
            labels.append(label)
    except:
        print(f"Disappear {path}")
        sys.stdout.flush()

    if len(imgs) == 128:

        imgs = torch.stack(imgs).cuda()
        with torch.no_grad():
            if split == 'val':
                features, gatings = model.forward_q(imgs)
                if len(global_gatings) ==0:
                    global_gatings = gatings
                else:
                    global_gatings = torch.cat((global_gatings,gatings))
            else:
                labels = torch.stack(labels).cuda()
                features = model.forward_p(imgs, labels)
                labels = []
            if len(global_features) == 0:
                global_features = features
            else:
                global_features = torch.cat((global_features,features))
        # print(len(imgs))
        imgs = []

imgs = torch.stack(imgs).cuda()
if split == 'trn':
    labels = torch.stack(labels).cuda()
with torch.no_grad():
    if split == 'val':
        features, gatings = model.forward_q(imgs)
        if len(global_gatings) ==0:
            global_gatings = gatings
        else:
            global_gatings = torch.cat((global_gatings,gatings))
    else:
        features = model.forward_p(imgs,labels)
    if len(global_features) == 0:
        global_features = features
    else:
        global_features = torch.cat((global_features,features))

features = global_features.cpu().numpy().astype(np.float32)
if split == 'val':
    gatings =  global_gatings.cpu().numpy().astype(np.float32)
    
save_file = os.path.join(save_dir, 'features_{}'.format(split))

if split == 'val':
    np.savez(save_file, examples=examples, features=features, gatings=gatings)
else:   
    np.savez(save_file, examples=examples, features=features)
