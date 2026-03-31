"""Based on https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer/blob/main/data/pascal.py
"""
import os
from PIL import Image
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset
from mae_utils import PURPLE, YELLOW
import json
import torchvision
def _read_img_rgb(path: str):
    im = Image.open(path).convert('RGB')
    return im.copy()

def _read_mask(path: str):
    im = Image.open(path)  
    return im.copy()


def create_grid_from_images_old(canvas, support_img, support_mask, query_img, query_mask):
   canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
   canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
   return canvas
 
class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, split, image_transform, mask_transform, padding: bool = 1, use_original_imgsize: bool = False, flipped_order: bool = False,
                reverse_support_and_query: bool = False, random: bool = False, ensemble: bool = False, purple: bool = False, cluster: bool = False, feature_name: str='features_vit_dino_val', percentage: str='', seed: int=0):
        self.fold = fold
        self.split = split
        self.nfolds = 4
        self.flipped_order = flipped_order
        self.nclass = 20 #20
        self.ncluster = 200
        self.padding = padding
        self.random = random
        self.ensemble = ensemble
        self.purple = purple
        self.cluster = cluster
        self.use_original_imgsize = use_original_imgsize
        self.datapath = datapath
        self.img_path = os.path.join(datapath, 'figures_dataset/pascal-5i/VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'figures_dataset/pascal-5i/VOC2012/SegmentationClassAug/')
        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform
 
        self.class_ids = self.build_class_ids()
        self.img_metadata_val = self.build_img_metadata('val') if '_val' in feature_name else self.build_img_metadata('trn') 
        self.img_metadata_trn = self.build_img_metadata('trn')
        self.feature_name = feature_name
        self.seed = seed
        self.percentage = percentage
        self.images_top50_val = self.get_top50_images_val()
        self.images_top50_trn = self.get_top50_images_trn()
         
 
    def __len__(self):
        return len(self.img_metadata_val)
 
    def get_top50_images_val(self):
        with open(f"{self.datapath}/figures_dataset/pascal-5i/VOC2012/{self.feature_name}_MoE_XY/folder{self.fold}_top50-similarity.json") as f:
            images_top50 = json.load(f)

        images_top50_new = {}
        for img_name, img_class in self.img_metadata_val:
            if img_name not in images_top50_new:
                images_top50_new[img_name] = {}

            images_top50_new[img_name]['top50'] = images_top50[img_name]
            images_top50_new[img_name]['class'] = img_class

        return images_top50_new

    def get_top50_images_trn(self):
        images_top50_new = {}
        for img_name, img_class in self.img_metadata_trn:
            if img_name not in images_top50_new:
                images_top50_new[img_name] = {}

            images_top50_new[img_name]['class'] = img_class

        return images_top50_new


    def create_grid_from_images(self, support_img, support_mask, query_img, query_mask, flip: bool = False):
        canvas = torch.ones((support_mask.shape[0], 2 * support_mask.shape[1] + 2 * self.padding, 2 * support_mask.shape[2] + 2 * self.padding))
        canvas[:, :support_mask.shape[1], -support_mask.shape[2]:] = support_mask
        canvas[:, -support_mask.shape[1]:, -support_mask.shape[2]:] = query_mask
        return canvas
 
    def __getitem__(self, idx):
        idx %= len(self.img_metadata_val)  # for testing, as n_images < 1000
        if self.cluster:
            query_name, class_sample_query, _ = self.img_metadata_val[idx]
        else:
            query_name, class_sample_query = self.img_metadata_val[idx]

        query_img_pil  = _read_img_rgb(os.path.join(self.img_path,  query_name) + '.jpg')
        query_mask_pil = _read_mask   (os.path.join(self.ann_path,  query_name) + '.png')

        if self.image_transform:
            query_img = self.image_transform(query_img_pil)
            query_mask_img, _ = self.extract_ignore_idx(query_mask_pil, class_sample_query, purple=self.purple)
            if self.mask_transform:
                query_mask = self.mask_transform(query_mask_img)
            else:
                query_mask = torchvision.transforms.ToTensor()(query_mask_img)
        else:
            raise RuntimeError("image_transform is needed!")

        grids = [] 

        for sim_idx in range(50):
            query_name2, support_name, class_sample_query2, class_sample_support = self.sample_episode(idx, sim_idx)

            support_img_pil  = _read_img_rgb(os.path.join(self.img_path, support_name) + '.jpg')
            support_mask_pil = _read_mask   (os.path.join(self.ann_path, support_name) + '.png')

            if self.image_transform:
                support_img = self.image_transform(support_img_pil)
            support_mask_img, _ = self.extract_ignore_idx(support_mask_pil, class_sample_support, purple=self.purple)
            if self.mask_transform:
                support_mask = self.mask_transform(support_mask_img)
            else:
                support_mask = torchvision.transforms.ToTensor()(support_mask_img)

            grid = self.create_grid_from_images(support_img, support_mask, query_img, query_mask, flip=self.flipped_order)
            grids.append(grid.unsqueeze(0))

        grid_stack = torch.vstack(grids) 
        return {'grid_stack': grid_stack}
 
    def extract_ignore_idx(self, mask, class_id, purple):
        mask = np.array(mask)
        boundary = np.floor(mask / 255.)
        if not purple:
            mask[mask != class_id + 1] = 0
            mask[mask == class_id + 1] = 255
            return Image.fromarray(mask), boundary
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x,y] != class_id + 1:
                    color_mask[x, y] = np.array(PURPLE)
                else:
                    color_mask[x, y] = np.array(YELLOW)
        return Image.fromarray(color_mask), boundary
    
    
    def load_frame(self, query_name, support_name):
        # import pdb;pdb.set_trace()
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_img = self.read_img(support_name)
        support_mask = self.read_mask(support_name)
        org_qry_imsize = query_img.size
    
        return query_img, query_mask, support_img, support_mask, org_qry_imsize
    
    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = Image.open(os.path.join(self.ann_path, img_name) + '.png')
        return mask
    
    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')
    
    def sample_episode(self, idx, sim_idx):
        """Returns the index of the query, support and class."""
        if self.cluster:
            query_name, class_sample, cluster_sample = self.img_metadata_val[idx]
        else:
            query_name, class_sample = self.img_metadata_val[idx]
    
        # import pdb;pdb.set_trace()
        if self.random:
            support_class = np.random.choice([k for k in self.img_metadata_classwise.keys() if self.img_metadata_classwise[k]], 1, replace=False)[0]
         
        support_name = self.images_top50_val[query_name]['top50'][sim_idx]
        support_class = self.images_top50_trn[support_name]['class']

        if support_name == query_name:
            print('support_name = query_name ' + support_name)
            return self.sample_episode(idx, sim_idx+1)
        

        return query_name, support_name, class_sample, support_class
    
    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        return class_ids_val
    
    def build_img_metadata(self,split):
    
        def read_metadata(split, fold_id):
            if self.cluster:
                fold_n_metadata_path = os.path.join(self.datapath, 'splits/pascal/%s/fold_cluster%d.txt' % (split, fold_id))
            else:
                fold_n_metadata_path = os.path.join(self.datapath, 'splits/pascal/%s/fold%d.txt' % (split, fold_id))
    
            with open(fold_n_metadata_path, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            # import pdb;pdb.set_trace()
            if self.cluster:
                fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1, int(data.split('__')[2]) - 1] for data in fold_n_metadata]
            else:
                fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            
            return fold_n_metadata
    
        img_metadata = []
        img_metadata = read_metadata(split, self.fold)
        
        print('Total (%s) images are : %d' % (split,len(img_metadata)))
    
        return img_metadata
    
    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []
    
        if len(self.img_metadata[0]) != 3:
            for img_name, img_class in self.img_metadata:
                img_metadata_classwise[img_class] += [img_name]
        else:
            for img_name, img_class, _ in self.img_metadata:
                img_metadata_classwise[img_class] += [img_name]
    
        return img_metadata_classwise
    

 
 
 

