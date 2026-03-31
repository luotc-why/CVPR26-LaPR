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
from_folder = sys.argv[7]
folder_ids = list(map(int, sys.argv[8].split(',')))


def extract_ignore_idx(mask, class_id):
    mask = np.array(mask)
    mask[mask != class_id + 1] = 0
    mask[mask == class_id + 1] = 255
    return Image.fromarray(mask)

def build_img_metadata(split, fold_id):
    fold_n_metadata_path = os.path.join(f'{data_base_path}/splits/pascal/{split}/fold{fold_id}.txt')

    with open(fold_n_metadata_path, 'r') as f:
        fold_n_metadata = f.read().split('\n')[:-1]
    # import pdb;pdb.set_trace()
    fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]    
    return fold_n_metadata

def get_mapper_name_class(fold_n_metadata):
    Mapper_name_class = {}
    for img_name, img_class in fold_n_metadata:
        if img_name not in Mapper_name_class:
            Mapper_name_class[img_name] = {}

        Mapper_name_class[img_name]['class'] = img_class

    return Mapper_name_class



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


save_dir = f"{data_base_path}/figures_dataset/pascal-5i/VOC2012/{feature_name}_{split}_MoE_final_XY_lr_{lr}_epoch_{checkpoint_epoch}_from_folder{from_folder}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    print(f"Directory exists at {save_dir}")
    # sys.exit()
# assert False

# meta_root =  f"/data1/liyusheng/CVPR-LRKL/Codes/evaluate/splits/pascal/{split}"
meta_root =  f"{data_base_path}/splits/pascal/{split}"
image_root = f"{data_base_path}/figures_dataset/pascal-5i/VOC2012/JPEGImages"
ann_root = f"{data_base_path}/figures_dataset/pascal-5i/VOC2012/SegmentationClassAug"

for folder_id in tqdm(folder_ids):
# for folder_id in tqdm([0, 1, 2, 3]):

    model = SupVitMLPMoE(pretrain_name,data_base_path=data_base_path)
    ckpt_path = f"{data_base_path}/save/final_seg_SupCon_moe_moe/path_moe_moe_models/SupCon_path_{pretrain_name}_folder_{from_folder}_seed_0_lr_{lr}_decay_0.0001_cropsz_{size}_bsz_64_temp_0.1_trial_0_cosine_pretrain-vit-freeze-encoder/{checkpoint_epoch}.pth"
    try:
        model.load_state_dict(clean_state_dict(torch.load(ckpt_path)['model']))
    except:
        print('{} is wrong'.format(ckpt_path))
        sys.stdout.flush()
        continue
    model.eval()
    model = model.cuda()
    print(f"Processing folder {folder_id}")
    sys.stdout.flush()
    with open(os.path.join(meta_root, 'fold'+str(folder_id)+'.txt')) as f:
        examples = f.readlines()
    if len(examples) == 0:
        print(f"zeros folder{folder_id}")
        sys.stdout.flush()
        continue
    image_names = [example.strip()[:-4] for example in examples]
    ann_examples = [os.path.join(ann_root, example.strip()[:-4]+'.png') for example in examples]
    examples = [os.path.join(image_root, example.strip()[:-4]+'.jpg') for example in examples]
       
    imgs = []

    if split == 'trn':
        labels = []
        fold_n_metadata = build_img_metadata(split='trn',fold_id=folder_id)
        Mapper_name_class = get_mapper_name_class(fold_n_metadata)
    
    global_features = torch.tensor([]).cuda()
    global_gatings = torch.tensor([]).cuda()
    
    for example, ann_example, image_name in zip(examples,ann_examples,image_names):
        try:
            path = os.path.join(example)
            img = Image.open(path).convert("RGB")
            img = center_crop(img)
            imgs.append(img)
            if split == 'trn':
                path_ann = os.path.join(ann_example)
                label_cls = Mapper_name_class[image_name]['class']
                label = Image.open(path_ann).convert('RGB')
                label = extract_ignore_idx(label,label_cls)
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
            print(len(imgs))
            imgs = []
    print(len(imgs))
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
    save_file = os.path.join(save_dir, 'folder'+str(folder_id))
    if split == 'val':
        np.savez(save_file, examples=examples, features=features, gatings=gatings)
    else:   
        np.savez(save_file, examples=examples, features=features)
