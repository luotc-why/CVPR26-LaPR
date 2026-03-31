import json
import os
import sys

feature_name = sys.argv[1]
output_name = sys.argv[2]
data_base_path = sys.argv[3]


inference_result_root = f'{data_base_path}/output_col_images/corr_3400_icl/'


cur_dir = os.path.join(inference_result_root, f'{output_name}_0/log.txt')

with open(cur_dir) as f:
    metas = f.readlines()
    
img_dir = f'{data_base_path}/figures_dataset/imagenet/train_data'
    
img_metadata = sorted(
    [f for f in os.listdir(img_dir)
    if os.path.isfile(os.path.join(img_dir, f))]
)

fold_n_metadata = [data[:-4]  for data in img_metadata]

with open(f'{data_base_path}/figures_dataset/imagenet/{feature_name}/new_top_50-similarity.json') as f:
    images_top50 = json.load(f)

iou_dict = {}
for cur_line in metas[1:-1]:
    img_id, sim_id, result = cur_line.split('\t')
    img_id, sim_id = int(img_id), int(sim_id)
    result = eval(result)
    iou = result['mse']
    image_name = fold_n_metadata[img_id]
    if image_name not in iou_dict:
        iou_dict[image_name] = {}
    # if image_name == images_top50[image_name][sim_id]:
        # print(images_top50[image_name])
        # print(image_name, images_top50[image_name][sim_id])
        # assert False
    iou_dict[image_name][images_top50[image_name][sim_id]] = iou
# delete the similarity of itself and then get the top5 and botton 5
# import pdb;pdb.set_trace()
for img_name in iou_dict:
    if img_name in iou_dict[img_name]:
        # print("big bug here")
        # assert False
        del iou_dict[img_name][img_name]
    # import pdb;pdb.set_trace()
    sorted_iou = sorted(iou_dict[img_name].items(), key=lambda x:x[1])
    sorted_iou_names = [x[0] for x in sorted_iou[:5]+sorted_iou[-5:]]
    iou_dict[img_name] = sorted_iou_names

save_dir = os.path.join(inference_result_root, f'{output_name}_0/contrastive.json')
os.makedirs(os.path.dirname(save_dir), exist_ok=True)
with open(save_dir,'w') as f:
    json.dump(iou_dict, f)
