import numpy as np
import scipy.spatial.distance as distance
import os
from tqdm import tqdm
import sys
from scipy import linalg, mat, dot
import json


features_name = sys.argv[1]
source_split = sys.argv[2]
target_split = sys.argv[3]
data_base_path = sys.argv[4]

print(f"Processing {features_name} ...")
sys.stdout.flush()
source_features_dir = f"{data_base_path}/figures_dataset/pascal-5i/VOC2012/{features_name}/"
target_features_dir = f"{data_base_path}/figures_dataset/pascal-5i/VOC2012/{features_name}/"
sys.stdout.flush()
source_path = os.path.join(source_features_dir, f'features_{source_split}.npz')
target_path = os.path.join(target_features_dir, f'features_{target_split}.npz')
try:
    source_file_npz = np.load(source_path)
    target_file_npz = np.load(target_path)
except:
    print(f"no folder {source_path} {target_path} ...")
    sys.stdout.flush()
    sys.exit()
source_examples = source_file_npz["examples"].tolist()
target_examples = target_file_npz["examples"].tolist()
source_features = source_file_npz["features"].astype(np.float32)
target_features = target_file_npz["features"].astype(np.float32)

target_sample_feature = target_features
similarity = dot(source_features,target_sample_feature.T)/(linalg.norm(source_features,axis=1, keepdims=True) * linalg.norm(target_sample_feature,axis=1, keepdims=True).T)

for i in range(len(similarity)):
    similarity[i][i] = 0

similarity_idx = np.argsort(similarity,axis=1)[:,-50:]

similarity_idx_dict = {}
for _, (cur_example, cur_similarity) in enumerate(zip(source_examples,similarity_idx)):
    img_name = cur_example.strip().split('/')[-1][:-4]

    cur_similar_name = list(target_examples[idx].strip().split('/')[-1][:-4] for idx in cur_similarity[::-1])
    cur_similar_name =  list(dict.fromkeys(cur_similar_name))

    assert len(cur_similar_name) >= 50, "num of cur_similar_name is too small, please enlarge the similarity_idx size"

    if img_name not in similarity_idx_dict:
        similarity_idx_dict[img_name] = list(target_examples[idx].strip().split('/')[-1][:-4]+' '+str(idx) for idx in cur_similarity[::-1])

with open(f"{source_features_dir}/fff_top50-similarity.json", "w") as outfile:
    json.dump(similarity_idx_dict, outfile)
    
