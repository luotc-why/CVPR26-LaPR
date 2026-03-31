import os
import io
import sys
from PIL import Image
import torch.utils.data as data
import cv2
import numpy as np
import json


NO_LABEL = -1

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']

# __all__ = ['contrastive_dataset']

def pil_loader(img_str):
    # buff = io.BytesIO(img_str)
    with Image.open(img_str) as img:
        img = img.convert('RGB')
    return img


def build_samples(img_root):
    return [f[:-4] for f in os.listdir(img_root) ]


class contrastive_dataset(data.Dataset):
    def __init__(self, data_base_path = '/data1/liyusheng/CVPR-LRKL/Data',img_root='/data1/liyusheng/CVPR-LRKL/Data/figures_dataset/pascal-5i/VOC2012/det_train_image', 
                 ann_root='/data1/liyusheng/CVPR-LRKL/Data/figures_dataset/pascal-5i/VOC2012/det_train_label', transform=None):
        
        # self.samples = []
        self.data_base_path = data_base_path
        self.img_root = img_root
        self.ann_root = ann_root
        contrastive_samples_root = f'{data_base_path}/output_det_images/corrert_vicl_performance'
        contrastive_samples_path = os.path.join(contrastive_samples_root, 'output_vit-laion2b-clip_trn_0/output_train_0/contrastive.json')
        contrastive_label_matching_root = f'{data_base_path}/output_det_images/label_matching'
        contrastive_label_matching_path = os.path.join(contrastive_label_matching_root, 'output_vit-laion2b-clip_trn_0/output_train_0/contrastive.json')

        with open(contrastive_samples_path) as f:
            self.contrastive_samples = json.load(f)
        
        with open(contrastive_label_matching_path) as f:
            self.contrastive_label_matching_samples = json.load(f)
        

        self.samples = build_samples(f"{data_base_path}/figures_dataset/pascal-5i/VOC2012/det_train_image")

        self.transform = transform
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img_name = self.samples[index]
        samples = []
        try:
            sample = pil_loader(os.path.join(self.img_root, img_name+'.png'))
        except:
            print(os.path.join(self.img_root, img_name+'.png'))

        samples.append(self.transform(sample))

        positive_sample_name = np.random.choice(self.contrastive_samples[img_name][:5])
        positive_sample = pil_loader(os.path.join(self.img_root, positive_sample_name+'.png'))
        samples.append(self.transform(positive_sample))

        positive_label = pil_loader(os.path.join(self.ann_root, positive_sample_name+'.png'))
        samples.append(self.transform(positive_label))

        negative_sample_name = np.random.choice(self.contrastive_samples[img_name][5:])
        negative_sample = pil_loader(os.path.join(self.img_root, negative_sample_name+'.png'))
        samples.append(self.transform(negative_sample))

        negative_label = pil_loader(os.path.join(self.ann_root, negative_sample_name+'.png'))
        samples.append(self.transform(negative_label))


        contrastive_positive_sample_name = np.random.choice(self.contrastive_label_matching_samples[img_name][:5])
        label_positive_sample = pil_loader(os.path.join(self.img_root, contrastive_positive_sample_name+'.png'))
        samples.append(self.transform(label_positive_sample))

        contrastive_positive_label = pil_loader(os.path.join(self.ann_root, contrastive_positive_sample_name+'.png'))
        samples.append(self.transform(contrastive_positive_label))

        contrastive_negative_sample_name = np.random.choice(self.contrastive_label_matching_samples[img_name][5:])
        label_negative_sample = pil_loader(os.path.join(self.img_root, contrastive_negative_sample_name+'.png'))
        samples.append(self.transform(label_negative_sample))

        contrastive_negative_label = pil_loader(os.path.join(self.ann_root, contrastive_negative_sample_name+'.png'))
        samples.append(self.transform(contrastive_negative_label))

        # if self.transform is not None:
        #     samples = self.transform(samples)

        return samples

    def __len__(self):
        # return 32
        return len(self.samples)
    