import os
import sys
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------- CLI ----------------------
features_name = sys.argv[1]
source_split = sys.argv[2]  # val q
target_split = sys.argv[3]  # trn p
data_base_path = sys.argv[4]
lr = sys.argv[5]
checkpoint_epoch = sys.argv[6]

print(f"Processing {features_name} ...")
sys.stdout.flush()

source_features_dir = f"{data_base_path}/figures_dataset/imagenet/{features_name}_col_lr_{lr}_epoch_{checkpoint_epoch}"  # val q
target_features_dir = f"{data_base_path}/figures_dataset/imagenet/{features_name}_col_lr_{lr}_epoch_{checkpoint_epoch}"  # trn p
source_path = os.path.join(source_features_dir, 'features_val.npz')  # val q
target_path = os.path.join(target_features_dir, 'features_trn.npz')  # trn p

try:
    source_file_npz = np.load(source_path, allow_pickle=True) 
    target_file_npz = np.load(target_path, allow_pickle=True)
except Exception:
    print("no folder feature_file ...")
    sys.exit(0)

source_examples = source_file_npz["examples"].tolist()
target_examples = target_file_npz["examples"].tolist()
source_features_np = source_file_npz["features"].astype(np.float32)  # [Nq, D]
target_features_np = target_file_npz["features"].astype(np.float32)  # [Np, K, D]
source_gatings_np  = source_file_npz["gatings"].astype(np.float32)   # [Nq, K]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print("WARNING: CUDA not available, running on CPU. (This will be slower.)")
torch.set_grad_enabled(False)

source_features = torch.from_numpy(source_features_np).to(device)  # [Nq, D]
target_features = torch.from_numpy(target_features_np).to(device)  # [Np, K, D]
source_gatings  = torch.from_numpy(source_gatings_np).to(device)   # [Nq, K]

K_t = target_features.shape[1]
K_g = source_gatings.shape[1]
assert K_t == K_g, f"K mismatch: target has {K_t}, but gatings has {K_g}"

src_unit = F.normalize(source_features, p=2, dim=1, eps=1e-12)  # [Nq, D]

similarity_idx_dict = {}
Np = target_features.shape[0]
topM = min(200, Np)

for i, cur_example in enumerate(tqdm(source_examples, total=len(source_examples), desc="Query-specific fusion")):
    g = source_gatings[i]  # [K]

    fused_t = (target_features * g.view(1, -1, 1)).sum(dim=1)
    fused_t = F.normalize(fused_t, p=2, dim=1, eps=1e-12)  

    sims = fused_t @ src_unit[i]  # [Np]

    if topM >= sims.shape[0]:
        _, top_idx = torch.sort(sims, descending=True)
    else:
        _, top_idx = torch.topk(sims, k=topM, largest=True, sorted=True)

    top_idx = top_idx.detach().tolist()

    img_name = cur_example.strip().split('/')[-1][:-5]
    seen = set()
    cur_similar_name = []
    for j in top_idx:
        name = target_examples[j].strip().split('/')[-1][:-4]
        if name not in seen:
            seen.add(name)
            cur_similar_name.append(name)

    assert len(cur_similar_name) >= 50, "num of cur_similar_name is too small, please enlarge the topM size"

    if img_name not in similarity_idx_dict:
        if source_split == target_split:
            similarity_idx_dict[img_name] = cur_similar_name[1:51]
        else:
            similarity_idx_dict[img_name] = cur_similar_name[:50]

with open(os.path.join(source_features_dir, "new_top50-similarity.json"), "w") as outfile:
    json.dump(similarity_idx_dict, outfile)
