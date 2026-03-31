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

def extract_ignore_idx(mask, class_id):
    mask = np.array(mask)
    mask[mask != class_id + 1] = 0
    mask[mask == class_id + 1] = 255
    return Image.fromarray(mask)

def build_img_metadata(data_base_path, split, fold_id):
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


class contrastive_dataset(data.Dataset):
    def __init__(self, data_base_path = '/data1/liyusheng/CVPR-LRKL/Data',img_root='/data1/liyusheng/CVPR-LRKL/Data/figures_dataset/pascal-5i/VOC2012/JPEGImages', 
                 ann_root='/data1/liyusheng/CVPR-LRKL/Data/figures_dataset/pascal-5i/VOC2012/SegmentationClassAug/',
                    meta='fold0.txt', folder_id=0, transform=None):
        
        # self.samples = []
        self.data_base_path = data_base_path
        self.img_root = img_root
        self.ann_root = ann_root
        self.classes = set()
        self.folder_id = folder_id
        meta = 'fold%d.txt' % (folder_id)
        self.fold_n_metadata = build_img_metadata(data_base_path=data_base_path,split='trn',fold_id=folder_id)
        self.Mapper_name_class = get_mapper_name_class(self.fold_n_metadata)
        contrastive_samples_root = f'{data_base_path}/output_seg_images/final_corrert_output_seg_images'
        contrastive_samples_path = os.path.join(contrastive_samples_root, 'output_vicl_performance_%d_0/contrastive.json' % (folder_id))
        contrastive_label_matching_root = f'{data_base_path}/output_seg_images/label_matching'
        contrastive_label_matching_path = os.path.join(contrastive_label_matching_root, 'output_vit-laion2b-clip_trn_%d_0/label_matching_contrastive.json' % (folder_id))

        with open(contrastive_samples_path) as f:
            self.contrastive_samples = json.load(f)
        
        with open(contrastive_label_matching_path) as f:
            self.contrastive_label_matching_samples = json.load(f)
        
        with open(os.path.join(f'{data_base_path}/splits/pascal/trn',meta)) as f:
            fold_n_metadata = f.read().split('\n')[:-1]

        self.samples = [data.split('__')[0] for data in fold_n_metadata]

        self.transform = transform
        
        self.initialized = False

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
            sample = pil_loader(os.path.join(self.img_root, img_name+'.jpg'))
        except:
            print(os.path.join(self.img_root, img_name+'.jpg'))

        samples.append(self.transform(sample))

        positive_sample_name = np.random.choice(self.contrastive_samples[img_name][:5])
        positive_sample = pil_loader(os.path.join(self.img_root, positive_sample_name+'.jpg'))
        samples.append(self.transform(positive_sample))

        positive_label = pil_loader(os.path.join(self.ann_root, positive_sample_name+'.png'))
        positive_label_cls = self.Mapper_name_class[positive_sample_name]['class']
        positive_label = extract_ignore_idx(positive_label,positive_label_cls)
        samples.append(self.transform(positive_label))

        negative_sample_name = np.random.choice(self.contrastive_samples[img_name][5:])
        negative_sample = pil_loader(os.path.join(self.img_root, negative_sample_name+'.jpg'))
        samples.append(self.transform(negative_sample))

        negative_label = pil_loader(os.path.join(self.ann_root, negative_sample_name+'.png'))
        negative_label_cls = self.Mapper_name_class[negative_sample_name]['class']
        negative_label = extract_ignore_idx(negative_label,negative_label_cls)
        samples.append(self.transform(negative_label))


        contrastive_positive_sample_name = np.random.choice(self.contrastive_label_matching_samples[img_name][:5])
        label_positive_sample = pil_loader(os.path.join(self.img_root, contrastive_positive_sample_name+'.jpg'))
        samples.append(self.transform(label_positive_sample))

        contrastive_positive_label = pil_loader(os.path.join(self.ann_root, contrastive_positive_sample_name+'.png'))
        contrastive_positive_label_cls = self.Mapper_name_class[contrastive_positive_sample_name]['class']
        contrastive_positive_label = extract_ignore_idx(contrastive_positive_label,contrastive_positive_label_cls)
        samples.append(self.transform(contrastive_positive_label))

        contrastive_negative_sample_name = np.random.choice(self.contrastive_label_matching_samples[img_name][5:])
        label_negative_sample = pil_loader(os.path.join(self.img_root, contrastive_negative_sample_name+'.jpg'))
        samples.append(self.transform(label_negative_sample))

        contrastive_negative_label = pil_loader(os.path.join(self.ann_root, contrastive_negative_sample_name+'.png'))
        contrastive_negative_label_cls = self.Mapper_name_class[contrastive_negative_sample_name]['class']
        contrastive_negative_label = extract_ignore_idx(contrastive_negative_label,contrastive_negative_label_cls)
        samples.append(self.transform(contrastive_negative_label))

        # if self.transform is not None:
        #     samples = self.transform(samples)

        return samples

    def __len__(self):
        # return 32
        return len(self.samples)
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of classes: {}\n'.format(len(self.classes))
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str