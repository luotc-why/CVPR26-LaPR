import os.path
from tqdm import trange
import pascal_dataloader_label_matching
from evaluate_detection.box_ops import to_rectangle
from evaluate_detection.canvas_ds import CanvasDataset
from reasoning_dataloader import *
import torchvision
from mae_utils import *
import argparse
from pathlib import Path
from evaluate.mae_utils import WHITE, YELLOW, PURPLE, BLACK

def _calc_metric_batch(ours_bhw3_u8: torch.Tensor,
                       target_bhw3_u8: torch.Tensor,
                       fg_rgb: tuple) -> torch.Tensor:
    assert ours_bhw3_u8.dtype == torch.uint8 and target_bhw3_u8.dtype == torch.uint8
    device = ours_bhw3_u8.device
    fg = torch.tensor(fg_rgb, dtype=torch.uint8, device=device).view(1, 1, 1, 3)
    seg_orig = (target_bhw3_u8 == fg).all(dim=-1)   # [B,H,W]
    seg_our  = (ours_bhw3_u8   == fg).all(dim=-1)   # [B,H,W]
    inter = (seg_orig & seg_our).sum(dim=(1, 2)).float()
    union = (seg_orig | seg_our).sum(dim=(1, 2)).float().clamp_min(1.0)
    return inter / union  # [B]


def get_args():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--output_dir', default='../output_dir/')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--base_dir', default='/data1/liyusheng/CVPR-LRKL/Data', help='pascal base dir')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--t', default=[0, 0, 0], type=float, nargs='+')
    parser.add_argument('--task', default='segmentation', choices=['segmentation', 'detection'])
    parser.add_argument('--ckpt', help='model checkpoint')
    parser.add_argument('--dataset_type', default='pascal',
                        choices=['pascal', 'pascal_det'])
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--split', default='val', type=str)
    parser.add_argument('--purple', default=0, type=int)
    parser.add_argument('--flip', default=0, type=int)
    parser.add_argument('--feature_name', default='features_supcon-in1k-pretrain_val', type=str)
    parser.add_argument('--percentage', default='', type=str)
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--random', action='store_true')
    return parser

def evaluate(args):
    log_path = os.path.join(args.output_dir, 'log.txt')

    with open(log_path, 'w') as log:
        log.write(str(args) + '\n')
    padding = 1
    image_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         torchvision.transforms.ToTensor()])
    mask_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         torchvision.transforms.ToTensor()])
    # import pdb;pdb.set_trace()
    # import pdb;pdb.set_trace()
    # ds = {
    #     'pascal': pascal_dataloader_label_matching.DatasetPASCAL,
    #     'pascal_det': CanvasDataset
    # }[args.dataset_type](args.base_dir, fold=args.fold, split=args.split, image_transform=image_transform, mask_transform=mask_transform,
    #                      flipped_order=args.flip, purple=args.purple, random=args.random, cluster=args.cluster, feature_name=args.feature_name, percentage=args.percentage, seed=args.seed)
    # Build the transforms:
    ds = CanvasDataset()
    device = torch.device(args.device)
    _mean = torch.as_tensor(imagenet_mean, device=device).view(3, 1, 1)
    _std  = torch.as_tensor(imagenet_std,  device=device).view(3, 1, 1)
    _mean = _mean.cuda()
    _std = _std.cuda()
    fg_rgb = YELLOW if args.purple else WHITE
    N = len(ds)

    eval_dict = {'iou': 0, 'color_blind_iou': 0, 'accuracy': 0}
    for idx in trange(len(ds)):
        sample = ds[idx]
        grid_stack = sample['grid_stack']
        #  list[Tensor] to [G,3,H,W] Tensor
        if isinstance(grid_stack, torch.Tensor):
            grids = grid_stack
        else:
            grids = torch.stack(grid_stack, dim=0)  # [G,3,H,W]
        grids = grids.cuda()
        if args.dataset_type == 'pascal_det':
            grids = grids * _std + _mean  
        
        imgs_u8 = (grids * 255.0).clamp(0, 255).round().to(torch.uint8)  # [G,3,H,W]
        imgs_u8 = imgs_u8.to(device, non_blocking=True).permute(0, 2, 3, 1).contiguous()  # [G,H,W,3]

        ours_b = imgs_u8[:, 113:, 113:, :]   # [G, H1, W1, 3]
        tgt_b  = imgs_u8[:,  :111, 113:, :]  # [G, H2, W2, 3]

        iou_b = _calc_metric_batch(ours_b, tgt_b, fg_rgb)  # [G]

        lines = []
        for g, iou in enumerate(iou_b.tolist()):
            lines.append(f"{idx}\t{g}\t{{'iou': {iou}, 'color_blind_iou': 0, 'accuracy': 0}}\n")
        with open(log_path, 'a') as log:
            log.writelines(lines)

        eval_dict['iou'] += (iou_b.sum().item() / N)

    with open(log_path, 'a') as log:
        log.write('all\t' + str(eval_dict) + '\n')


if __name__ == '__main__':
    args = get_args()

    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate(args)
