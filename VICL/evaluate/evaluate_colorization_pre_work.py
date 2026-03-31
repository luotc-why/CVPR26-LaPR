import sys
import os
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cwd))
import os.path

import torchvision
from tqdm import trange
from evaluate.in_colorization_dataloader import DatasetColorization
from evaluate.reasoning_dataloader import *
from evaluate.mae_utils import *
import argparse
from pathlib import Path
import time

def _generate_result_for_canvas_batch(args, model, canvases_bchw):
    B, C, H, W = canvases_bchw.shape

    ids_shuffle, len_keep = generate_mask_for_evaluation()
    ids_shuffle = ids_shuffle.to(args.device)
    if ids_shuffle.dim() == 1:  # (L,) -> (B,L)
        ids_shuffle = ids_shuffle.unsqueeze(0).repeat(B, 1)
    ids_shuffle = ids_shuffle.repeat(B, 1)

    # print(ids_shuffle.shape,'ids_shuffle.shape')
    _, im_paste, _ = generate_image_for_batch(
        canvases_bchw.to(args.device), model, ids_shuffle, len_keep, device=args.device
    ) 

    mean = torch.as_tensor(imagenet_mean, dtype=canvases_bchw.dtype, device=canvases_bchw.device)[None, :, None, None]
    std = torch.as_tensor(imagenet_std, dtype=canvases_bchw.dtype, device=canvases_bchw.device)[None, :, None, None]
    originals = (canvases_bchw * std + mean)  # (B,C,H,W) in [0,1] scale *should* be
    originals = torch.einsum('bchw->bhwc', originals) * 255.0
    originals = torch.clamp(originals, 0, 255).to(torch.uint8).cpu().numpy()  # (B,H,W,3) uint8

    if isinstance(im_paste, torch.Tensor):
        gens = im_paste.detach().to(torch.uint8).cpu().numpy()
    else:
        gens = im_paste.astype(np.uint8)
    assert originals.shape == gens.shape, (originals.shape, gens.shape)

    return [originals[i] for i in range(B)], [gens[i] for i in range(B)]


def get_args():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--model', default='mae_vit_small_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--output_dir', default='../output_dir/')
    parser.add_argument('--data_path')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--tta_option', default=0, type=int)
    parser.add_argument('--ckpt', help='resume from checkpoint')
    parser.add_argument('--meta_split', default='0', help='meta_split')
    parser.add_argument('--feature_name', default='features_vit_val', help='meta_split')
    parser.add_argument('--sim_batch', default=25, type=int, help='batch size for similarity evaluation')
    parser.add_argument('--part', type=int, default=1, help='index of this part, from 0 to 4')
    parser.add_argument('--num_parts', type=int, default=20, help='total number of parts')
    parser.set_defaults(autoregressive=False)
    return parser


def _generate_result_for_canvas(args, model, canvas):
    """canvas is already in the right range."""
    ids_shuffle, len_keep = generate_mask_for_evaluation()
    _, im_paste, _ = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                    len_keep, device=args.device)
    canvas = torch.einsum('chw->hwc', canvas)
    canvas = torch.clip((canvas.cpu().detach() * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy()
    assert canvas.shape == im_paste.shape, (canvas.shape, im_paste.shape)
    return np.uint8(canvas), np.uint8(im_paste)


def calculate_metric(args, target, ours):
    ours = (np.transpose(ours/255., [2, 0, 1]) - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
    target = (np.transpose(target/255., [2, 0, 1]) - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

    target = target[:, 113:, 113:]
    ours = ours[:, 113:, 113:]
    mse = np.mean((target - ours)**2)
    return {'mse': mse}

def _as_list_of_canvases(grid):
    if isinstance(grid, torch.Tensor) and grid.dim() == 4:
        return [grid[i] for i in range(grid.shape[0])]
    elif isinstance(grid, (list, tuple)):
        return list(grid)
    else:
        raise TypeError(f'Unexpected grid type/shape: {type(grid)}')

def convert_to_rgb(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def evaluate(args):
    with open(os.path.join(args.output_dir, f'log_part{args.part}.txt'), 'w') as log:
        log.write(str(args) + '\n')
    print(args.output_dir)
    model = prepare_model(args.ckpt, arch=args.model, vq_base_dir=args.data_path)
    model = model.to(args.device)
    padding = 1

    image_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         convert_to_rgb,
         torchvision.transforms.ToTensor()])
    mask_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         convert_to_rgb,
         torchvision.transforms.ToTensor()])
    ds = DatasetColorization(args.data_path, image_transform, mask_transform, split = args.meta_split, feature_name=args.feature_name, seed=args.seed)

    eval_dict = {'mse': 0.}
    SIM_TOTAL = 50
    B = args.sim_batch
    total_len = len(ds)
    part_size = total_len // args.num_parts
    start_idx = args.part * part_size
    end_idx = min((args.part + 1) * part_size, total_len)
    print(f'Processing part {args.part}: idx [{start_idx}, {end_idx})')
    for idx in trange(start_idx, end_idx):
        grid = _as_list_of_canvases(ds[idx]['grids'])  
        assert len(grid) == SIM_TOTAL, f'Expect {SIM_TOTAL}, got {len(grid)}'

        for start in range(0, SIM_TOTAL, B):
            chunk_idx = list(range(start, min(start+B, SIM_TOTAL)))
            canvases_b = torch.stack([
                (grid[s] - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
                for s in chunk_idx
            ], dim=0).to(args.device)

            originals_b, gens_b = _generate_result_for_canvas_batch(args, model, canvases_b)

            for local_i, sim_idx in enumerate(chunk_idx):
                original_image = originals_b[local_i]
                generated_result = gens_b[local_i]
                # print(f'Processing image {idx}, sim idx {sim_idx}')
                current_metric = calculate_metric(args, original_image, generated_result)
                with open(os.path.join(args.output_dir, f'log_part{args.part}.txt'), 'a') as log:
                    log.write(f'{idx}\t{sim_idx}\t{current_metric}\n')
                for k, v in current_metric.items():
                    eval_dict[k] += (v / len(ds))  

    with open(os.path.join(args.output_dir, f'log_part{args.part}.txt'), 'a') as log:
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