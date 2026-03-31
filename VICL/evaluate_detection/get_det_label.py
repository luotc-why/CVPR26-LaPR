import torch.utils.data as data
import sys
print(sys.path)
sys.path.append('..')
# from evaluate_detection.voc_orig import VOCDetection as VOCDetectionOrig
# from evaluate_detection.voc_orig import VOCDetection as VOCDetectionOrig
from voc_orig import VOCDetection as VOCDetectionOrig
from voc_orig import VOCDetection as VOCDetectionOrig

import cv2
from PIL import Image
from matplotlib import pyplot as plt
import torch
import numpy as np
import torchvision.transforms as T
import json
import tqdm
def box_to_img(mask, target, border_width=4):
    if mask is None:
        mask = np.zeros((112, 112, 3))
    h, w, _ = mask.shape
    for box in target['boxes']:
        x_min, y_min, x_max, y_max = list((box * (h - 1)).round().int().numpy())
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), (255, 255, 255), border_width)
    return Image.fromarray(mask.astype('uint8'))


def get_annotated_image(img, boxes, border_width=3, mode='draw', bgcolor='white', fg='image'):
    if mode == 'draw':
        image_copy = np.array(img.copy())
        for box in boxes:
            box = box.numpy().astype('int')
            cv2.rectangle(image_copy, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), border_width)
    elif mode == 'keep':
        image_copy = np.array(Image.new('RGB', (img.shape[1], img.shape[0]), color=bgcolor))

        for box in boxes:
            box = box.numpy().astype('int')
            if fg == 'image':
                image_copy[box[1]:box[3], box[0]:box[2]] = img[box[1]:box[3], box[0]:box[2]]
            elif fg == 'white':
                image_copy[box[1]:box[3], box[0]:box[2]] = 255
    return image_copy




# ids_shuffle, len_keep = generate_mask_for_evaluation_2rows()

class CanvasDataset(data.Dataset):

    def __init__(self, pascal_path='/data1/liyusheng/CVPR-LRKL/Data/figures_dataset/pascal-5i', top_50_path = "/data1/liyusheng/CVPR-LRKL/Data/figures_dataset/pascal-5i/VOC2012/features_vit-laion2b_det/fff_top50-similarity.json", years=("2012",), random=False, feature_name='features_rn50_val_det', **kwargs):
        self.train_ds = VOCDetectionOrig(pascal_path, years, image_sets=['train'], transforms=None)
        self.val_ds = VOCDetectionOrig(pascal_path, years, image_sets=['val'], transforms=None)
        self.background_transforms = T.Compose([
            T.Resize((224, 224)),
            T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ])
        self.feature_name = feature_name
        self.random = random

    def __len__(self):
        # return len(self.train_ds)
        return len(self.val_ds)

    def __getitem__(self, idx):
        
        query_image, query_target = self.val_ds[idx]
        query_image_name = self.val_ds.images[idx].split('/')[-1][:-4]
        label = np.random.choice(query_target['labels']).item()
        boxes = query_target['boxes'][torch.where(query_target['labels'] == label)[0]]
        query_image_copy = get_annotated_image(np.array(query_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
        query_image_copy_pil = Image.fromarray(query_image_copy)

        # save for visualization
        query_image.save(f'/data1/liyusheng/CVPR-LRKL/Data/figures_dataset/pascal-5i/VOC2012/det_test_image/{query_image_name}.png')
        query_image_copy_pil.save(f'/data1/liyusheng/CVPR-LRKL/Data/figures_dataset/pascal-5i/VOC2012/det_test_label/{query_image_name}.png')

        return 

if __name__ == "__main__":

    canvas_ds = CanvasDataset()

    for idx in range(len(canvas_ds)):
        canvas_ds[idx]