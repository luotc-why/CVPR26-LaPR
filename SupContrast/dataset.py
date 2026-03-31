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

class contrastive_dataset(data.Dataset):
    def __init__(self, img_root='/data1/liyusheng/CVPR-LRKL/Data/figures_dataset/pascal-5i/VOC2012/JPEGImages', 
                    meta='fold0.txt', folder_id=0, transform=None):
        
        # self.samples = []
        self.img_root = img_root
        self.classes = set()
        self.folder_id = folder_id
        meta = 'fold%d.txt' % (folder_id)

        contrastive_samples_root = '/data1/liyusheng/CVPR-LRKL/Data/output_seg_images'
        contrastive_samples_path = os.path.join(contrastive_samples_root, 'output_vit-laion2b-clip_trn_%d_0/contrastive.json' % (folder_id))

        with open(contrastive_samples_path) as f:
            self.contrastive_samples = json.load(f)
        
        with open(os.path.join('/data1/liyusheng/CVPR-LRKL/Data/splits/pascal/trn',meta)) as f:
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

        negative_sample_name = np.random.choice(self.contrastive_samples[img_name][5:])
        negative_sample = pil_loader(os.path.join(self.img_root, negative_sample_name+'.jpg'))
        samples.append(self.transform(negative_sample))

        # if self.transform is not None:
        #     samples = self.transform(samples)

        return samples

    def __len__(self):
        return len(self.samples)
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of classes: {}\n'.format(len(self.classes))
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str