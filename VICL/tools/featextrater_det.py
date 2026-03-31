
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

import timm

model_name = sys.argv[1]
feature_name = sys.argv[2]
data_base_path = sys.argv[3]

sys.path.append('/data1/liyusheng/CVPR-LRKL/Codes/VICL')
from evaluate_detection.voc_orig import VOCDetection 

pretrained_cfg = timm.models.create_model('vit_large_patch14_clip_224').default_cfg
pretrained_cfg['file'] = f'{data_base_path}/weights/visual_prompt_retrieval/tools/vit_large_patch14_clip_224.laion2b/open_clip_pytorch_model.bin'
model = timm.create_model(model_name, pretrained=True,pretrained_cfg=pretrained_cfg)
model.eval()
model = model.cuda()

# import pdb;pdb.set_trace()

# load the image transformer
t = []
# maintain same ratio w.r.t. 224 images
# follow https://github.com/facebookresearch/mae/blob/main/util/datasets.py
t.append(T.Resize(model.pretrained_cfg['input_size'][1], interpolation=Image.BICUBIC))
t.append(T.CenterCrop(model.pretrained_cfg['input_size'][1]))
t.append(T.ToTensor())
t.append(T.Normalize(model.pretrained_cfg['mean'], model.pretrained_cfg['std']))
center_crop = T.Compose(t)


save_dir = f"{data_base_path}/figures_dataset/pascal-5i/VOC2012/{feature_name}_det"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    print(f"Directory exists at {save_dir}")
    sys.exit()

for cur_split in ['train','val']:
    ds = VOCDetection(f'{data_base_path}/figures_dataset/pascal-5i/', ['2012'], image_sets=[cur_split], transforms=None)

    global_features = torch.tensor([]).cuda()
    imgs = []
    examples = []
    for index in tqdm(range(len(ds))):
        try:
            example = ds.images[index]
            # import pdb;pdb.set_trace()
            img = Image.open(example).convert('RGB')
            examples.append(example)
            img = center_crop(img)
            imgs.append(img)
        except:
            print(f"Disappear {ds.images[index]}")
            sys.stdout.flush()

        if len(imgs) == 128:

            imgs = torch.stack(imgs).cuda()
            # import pdb;pdb.set_trace()
            with torch.no_grad():
                features = model.forward_features(imgs)
                features = model.forward_head(features,pre_logits=True)
                if len(global_features) == 0:
                    global_features = features
                else:
                    global_features = torch.cat((global_features,features))

            imgs = []

    if len(imgs) > 0:
        imgs = torch.stack(imgs).cuda()
        with torch.no_grad():
            features = model.forward_features(imgs)
            features = model.forward_head(features,pre_logits=True)
            if len(global_features) == 0:
                global_features = features
            else:
                # import pdb;pdb.set_trace()
                global_features = torch.cat((global_features,features))

    features = global_features.cpu().numpy().astype(np.float32)
    save_file = os.path.join(save_dir, 'features_{}'.format(cur_split))
    np.savez(save_file, examples=examples, features=features)