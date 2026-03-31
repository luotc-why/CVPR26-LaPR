import json
import os
import sys
sys.path.append('/data1/liyusheng/CVPR-LRKL/Codes/VICL')
from evaluate_detection.voc_orig import VOCDetection 

inference_result_root = '/data1/liyusheng/CVPR-LRKL/Codes/VICL/output_det_images/output_vit-laion2b-clip_trn_0'

ds = VOCDetection('/data1/liyusheng/CVPR-LRKL/Data/figures_dataset/pascal-5i/', ['2012'], image_sets=['train'], transforms=None)

with open('/data1/liyusheng/CVPR-LRKL/Data/figures_dataset/pascal-5i/VOC2012/features_vit-laion2b_det/fff_top50-similarity.json') as f:
    images_top50 = json.load(f)

with open('/data1/liyusheng/CVPR-LRKL/Codes/VICL/output_det_images/output_vit-laion2b-clip_trn_0/log.txt') as f:
    metas = f.readlines()

iou_dict = {}
for cur_line in metas[1:-1]:
    # import pdb;pdb.set_trace()
    img_id, sim_id, result = cur_line.split('\t')
    img_id, sim_id = int(img_id), int(sim_id)
    result = eval(result)
    iou = result['iou']
    image_name = ds.images[img_id]
    image_name = image_name.split('/')[-1][:-4]
    if image_name not in iou_dict:
        iou_dict[image_name] = {}
    iou_dict[image_name][images_top50[image_name][sim_id]] = iou

# delete the similarity of itself and then get the top5 and botton 5
# import pdb;pdb.set_trace()
for img_name in iou_dict:
    if img_name in iou_dict[img_name]:
        del iou_dict[img_name][img_name]
        print("wrong!")
    # import pdb;pdb.set_trace()
    sorted_iou = sorted(iou_dict[img_name].items(), key=lambda x:x[1], reverse=True)
    sorted_iou_names = [x[0].split(' ')[0] for x in sorted_iou[:5]+sorted_iou[-5:]]
    iou_dict[img_name] = sorted_iou_names

save_dir = os.path.join(inference_result_root, 'output_train_0/contrastive.json')

os.makedirs(os.path.dirname(save_dir), exist_ok=True)

with open(save_dir,'w') as f:
    json.dump(iou_dict, f)



