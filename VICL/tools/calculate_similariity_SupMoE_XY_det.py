import numpy as np
import scipy.spatial.distance as distance
import os
from tqdm import tqdm
import sys
from scipy import linalg, mat, dot
import json


features_name = sys.argv[1]
source_split = sys.argv[2] #val q
target_split = sys.argv[3] #trn p
data_base_path = sys.argv[4]
lr = sys.argv[5]
checkpoint_epoch = sys.argv[6]

print(f"Processing {features_name} ...")
sys.stdout.flush()
source_features_dir = f"{data_base_path}/figures_dataset/pascal-5i/VOC2012/{features_name}_det_lr_{lr}_epoch_{checkpoint_epoch}" #val q
target_features_dir = f"{data_base_path}/figures_dataset/pascal-5i/VOC2012/{features_name}_det_lr_{lr}_epoch_{checkpoint_epoch}" #trn p
sys.stdout.flush()
source_path = os.path.join(source_features_dir, 'features_val.npz') #val q
target_path = os.path.join(target_features_dir, 'features_train.npz') #trn p
try:
    source_file_npz = np.load(source_path) #val q
    target_file_npz = np.load(target_path) #trn p
except:
    print(f"no folder feature_file ...")
    sys.exit()
source_examples = source_file_npz["examples"].tolist()
target_examples = target_file_npz["examples"].tolist()
source_features = source_file_npz["features"].astype(np.float32) # val q [Nq,D]
target_features = target_file_npz["features"].astype(np.float32) # trn p [Np,K=10,D]
source_gatings = source_file_npz["gatings"].astype(np.float32) # val q [Nq,K=10]

assert target_features.shape[1] == source_gatings.shape[1], \
    f"K mismatch: target has {target_features.shape[1]}, but gatings has {source_gatings.shape[1]}"
src_unit = source_features / (np.linalg.norm(source_features, axis=1, keepdims=True))
similarity_idx_dict = {}
topM = min(200, target_features.shape[0])  
for i, (cur_example, g) in enumerate(tqdm(zip(source_examples, source_gatings),
                                            total=len(source_examples),
                                            desc="Query-specific fusion")):
    g = g.astype(np.float32)
    fused_t = np.tensordot(target_features, g, axes=([1], [0])).astype(np.float32)
    fused_t /= (np.linalg.norm(fused_t, axis=1, keepdims=True))
    sims = np.dot(src_unit[i], fused_t.T)  # [Np]
    if topM >= sims.shape[0]:
        top_idx = np.argsort(sims)[::-1]
    else:
        part = np.argpartition(sims, -topM)[-topM:]
        top_idx = part[np.argsort(sims[part])[::-1]]
    img_name = cur_example.strip().split('/')[-1][:-4]
    cur_similar_name = [target_examples[j].strip().split('/')[-1][:-4] for j in top_idx]
    cur_similar_name = list(dict.fromkeys(cur_similar_name))
    assert len(cur_similar_name) >= 50, "num of cur_similar_name is too small, please enlarge the topM size"
    if img_name not in similarity_idx_dict:
        similarity_idx_dict[img_name] = list((f+' '+str(top_idx[i]) ) for i,f in enumerate(cur_similar_name[:50]))
        # similarity_idx_dict[img_name] = cur_similar_name[:50]

with open(f"{source_features_dir}/LaPR_top50-similarity.json", "w") as outfile:
    json.dump(similarity_idx_dict, outfile)
