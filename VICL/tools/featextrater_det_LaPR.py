
"""
Extract features using PlacesCNN.
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
from collections import OrderedDict
from VitMoE_XY import SupVitMLPMoE

import timm

def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.','') if 'module.' in k else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


t = []
size = 224
t.append(T.Resize((size,size), interpolation=Image.BICUBIC))
t.append(T.CenterCrop(size))
t.append(T.ToTensor())
t.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
center_crop = T.Compose(t)

model_name = sys.argv[1]
feature_name = sys.argv[2]
data_base_path = sys.argv[3]
lr = sys.argv[4]
checkpoint_epoch = sys.argv[5]

sys.path.append('/data1/liyusheng/CVPR-LRKL/Codes/VICL')
from evaluate_detection.voc_orig import VOCDetection 

model = SupVitMLPMoE(model_name,data_base_path=data_base_path)
ckpt_path = f"{data_base_path}/save/final_det_SupCon_moe_moe/path_moe_moe_models/SupCon_path_{model_name}_folder_0_seed_0_lr_{lr}_decay_0.0001_cropsz_{size}_bsz_64_temp_0.1_trial_0_cosine_pretrain-vit-freeze-encoder/{checkpoint_epoch}.pth"
try:
    model.load_state_dict(clean_state_dict(torch.load(ckpt_path)['model']))
except:
    print('{} is wrong'.format(ckpt_path))
    sys.exit()
model.eval()
model = model.cuda()
# import pdb;pdb.set_trace()

save_dir = f"{data_base_path}/figures_dataset/pascal-5i/VOC2012/{feature_name}_det_lr_{lr}_epoch_{checkpoint_epoch}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    print(f"Directory exists at {save_dir}")
    # sys.exit()

for cur_split in ['train','val']:
    ds = VOCDetection(f'{data_base_path}/figures_dataset/pascal-5i/', ['2012'], image_sets=[cur_split], transforms=None)

    global_features = torch.tensor([]).cuda()
    global_gatings = torch.tensor([]).cuda()
    imgs = []
    examples = []
    if cur_split == 'train':
        labels = []
        ann_root = f"{data_base_path}/figures_dataset/pascal-5i/VOC2012/det_train_label"
    for index in tqdm(range(len(ds))):
        try:
            example = ds.images[index]
            # import pdb;pdb.set_trace()
            img = Image.open(example).convert('RGB')
            examples.append(example)
            img = center_crop(img)
            imgs.append(img)
            if cur_split == 'train':
                ann_path = os.path.join(ann_root, example.split('/')[-1].replace('.jpg','.png'))
                cur_label = Image.open(ann_path).convert('RGB')
                cur_label = center_crop(cur_label)
                labels.append(cur_label)
        except:
            print(f"Disappear {ds.images[index]}")
            sys.stdout.flush()

        if len(imgs) == 128:

            imgs = torch.stack(imgs).cuda()
            # import pdb;pdb.set_trace()
            with torch.no_grad():
                if cur_split == 'val':
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

            imgs = []

    if len(imgs) > 0:
        imgs = torch.stack(imgs).cuda()
        labels = torch.stack(labels).cuda() if cur_split == 'train' else None
        with torch.no_grad():
            if cur_split == 'val':
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
                # import pdb;pdb.set_trace()
                global_features = torch.cat((global_features,features))

    features = global_features.cpu().numpy().astype(np.float32)
    if cur_split == 'val':
        gatings =  global_gatings.cpu().numpy().astype(np.float32)

    save_file = os.path.join(save_dir, 'features_{}'.format(cur_split))
    if cur_split == 'val':
        np.savez(save_file, examples=examples, features=features, gatings=gatings)
    else:   
        np.savez(save_file, examples=examples, features=features)
