import torch.utils.data as data
import sys
print(sys.path)
sys.path.append('..')
from evaluate_detection.voc_orig import VOCDetection as VOCDetectionOrig, make_transforms
import cv2
from PIL import Image
from evaluate_detection.voc import make_transforms
from matplotlib import pyplot as plt
import torch
import numpy as np
import torchvision.transforms as T
import json

def create_grid_from_images(canvas, support_img, support_mask, query_img, query_mask):
   canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
   canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
   canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
   canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
   return canvas


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
        self.transforms = make_transforms('val')
        self.random = random
        self.top_50_path = top_50_path
        self.images_top50 = self.get_top50_images()

    def get_top50_images(self):
        with open(self.top_50_path) as f:
            images_top50 = json.load(f)

        return images_top50


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
        grid_stack = torch.tensor([]).cuda()

        query_image_ten = self.transforms(query_image, None)[0]
        query_target_ten = self.transforms(query_image_copy_pil, None)[0]

        for sim_idx in range(1):
            
            _, support_image_idx = self.images_top50[query_image_name][sim_idx].split(' ')
            # print(f"support_image_idx: {support_image_idx}")
            support_image, support_target = self.train_ds[int(support_image_idx)]

            ### commend this if uncommend follow
            support_label = np.random.choice(support_target['labels']).item()
            boxes = support_target['boxes'][torch.where(support_target['labels'] == support_label)[0]]

            # boxes = support_target['boxes'][torch.where(support_target['labels'] == label)[0]]
            support_image_copy = get_annotated_image(np.array(support_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
            support_image_copy_pil = Image.fromarray(support_image_copy)

            support_target_ten = self.transforms(support_image_copy_pil, None)[0]
            support_image_ten = self.transforms(support_image, None)[0]

            background_image = Image.new('RGB', (224, 224), color='white')
            background_image = self.background_transforms(background_image)
            canvas = create_grid_from_images(background_image, support_image_ten, support_target_ten, query_image_ten,
                                            query_target_ten)
            canvas = canvas.unsqueeze(0)

            if len(grid_stack) == 0:
                grid_stack = canvas
            else:
                grid_stack = torch.cat((grid_stack,canvas))

        return {'grid_stack': grid_stack}


# if __name__ == "__main__":
#     # model = prepare_model('/shared/amir/Deployment/arxiv_mae/logs_dir/pretrain_small_arxiv2/checkpoint-799.pth',
#     #                       arch='mae_vit_small_patch16')

#     canvas_ds = CanvasDataset()

#     canvas = canvas_ds[(611)]