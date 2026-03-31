from __future__ import print_function

import os
import sys
import argparse
import time
import math
from torch.utils.tensorboard import SummaryWriter

# import tensorboard_logger as tb_logger   tensorflow no no no
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.SupVitMLPMoE import SupVitMLPMoE, set_train_parts_moe
from networks.resnet import SupConResNet

from losses import SupConLoss, RetriverConLoss
from dataset_tmoe_det import contrastive_dataset
import numpy as np

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=5,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--seed', type=int, default=0)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default= 'vit_large_patch14_clip_224.laion2b')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, default='(0.485, 0.456, 0.406)', help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, default='(0.229, 0.224, 0.225)', help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default='/data1/liyusheng/CVPR-LRKL/Data/figures_dataset/pascal-5i/VOC2012/JPEGImages', help='path to custom dataset')
    parser.add_argument('--meta', type=str, default='fold0.txt', help='path to meta file')
    parser.add_argument('--folder_id', type=int, default=0, help='path to meta file')
    parser.add_argument('--data_base_path',type=str, default='/data1/liyusheng/CVPR-LRKL/Data', help='base path to data')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')
    parser.add_argument('--pretrain', action='store_true', help='load official imagenet pre-trained')
    parser.add_argument('--feat_dim', type=int, default=128, help='feat dim')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = f'{opt.data_base_path}/save/final_det_SupCon_moe_moe/{opt.dataset}_moe_moe_models'
    opt.tb_path = f'{opt.data_base_path}/save/final_det_SupCon_moe_moe/{opt.dataset}_moe_moe_tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_folder_{}_seed_{}_lr_{}_decay_{}_cropsz_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.folder_id, opt.seed, opt.learning_rate,
               opt.weight_decay, opt.size, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.pretrain:
        opt.model_name = '{}_pretrain-vit-freeze-encoder'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        ann_root = os.path.join(opt.data_base_path, 'figures_dataset/pascal-5i/VOC2012/det_train_label')
        train_dataset = contrastive_dataset(data_base_path=opt.data_base_path,img_root=opt.data_folder, ann_root=ann_root, transform=train_transform)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
  
    model = SupVitMLPMoE(name=opt.model, data_base_path= opt.data_base_path)

    model.freeze_stages()

    # if opt.pretrain:
        # import pdb;pdb.set_trace()
        # model.encoder.load_state_dict(torch.load('/mnt/lustre/yhzhang/weights/resnet/resnet50_a1_0-14fe96d1.pth'), strict=False)
        # model.load_state_dict(torch.load('/mnt/lustre/yhzhang/SupContrast/weights/supcon.pth',map_location=torch.device('cpu'))['model_ema'])
      
    criterion = RetriverConLoss(temperature=opt.temp)
    # criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def get_loaded_loss(p_soft, eps=1e-12):
    B, K = p_soft.shape
    p_mean = p_soft.mean(dim=0).clamp_min(eps)             # (K,)
    u = p_mean.new_full((K,), 1.0 / K)
    # print(p_mean)
    loss = (p_mean * (p_mean.log() - math.log(1.0 / K))).sum()
    return loss


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    #freeze mode,
    model.freeze_stages()
    

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_vicl_performance = AverageMeter()
    losses_label_matching = AverageMeter()
    losses_loaded_avg = AverageMeter()
    end = time.time()
    for idx, (images) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # import pdb;pdb.set_trace()
        bsz = images[0].shape[0]
        if torch.cuda.is_available():
            images = [img.cuda(non_blocking=True) for img in images]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        
        
        # ===== Step 1：train mlps =====
        set_train_parts_moe(model, True, False)   
        optimizer.zero_grad(set_to_none=True)

        f1, f2, f3,  _ = model.constrative_forward(images[0], images[1], images[2], images[3], images[4])
        features_vicl_performance = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
        loss_vicl_performance = criterion(features_vicl_performance, labels=None) if opt.method=='SupCon' \
                                else criterion(features_vicl_performance)

        loss_vicl_performance.backward()
        optimizer.step()
        losses_vicl_performance.update(loss_vicl_performance.item(), bsz)

        # ===== Step 2：train router =====
        set_train_parts_moe(model, False, True)  
        optimizer.zero_grad(set_to_none=True)

        f1, f4, f5, gating_weights = model.constrative_forward(images[0], images[5], images[6], images[7], images[8])
        features_label_matching = torch.cat([f1.unsqueeze(1), f4.unsqueeze(1), f5.unsqueeze(1)], dim=1)
        loss_label_matching = criterion(features_label_matching, labels=None) if opt.method=='SupCon' \
                            else criterion(features_label_matching)
        loss_loaded_avg = get_loaded_loss(gating_weights)
        loss_label = loss_label_matching + 0.5 * loss_loaded_avg

        loss_label.backward()
        optimizer.step()
        losses_label_matching.update(loss_label_matching.item(), bsz)
        losses_loaded_avg.update(loss_loaded_avg.item(), bsz)

        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % 1 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss_vicl_performance {loss_vicl_performance.val:.3f} ({loss_vicl_performance.avg:.3f})\t'
                  'loss_label_matching {loss_label_matching.val:.3f} ({loss_label_matching.avg:.3f})\t'
                  'losses_loaded_avg {losses_loaded_avg.val:.3f} ({losses_loaded_avg.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_vicl_performance=losses_vicl_performance,loss_label_matching=losses_label_matching,losses_loaded_avg=losses_loaded_avg))
            sys.stdout.flush()

    return losses_vicl_performance.avg,losses_label_matching.avg,losses_loaded_avg.avg


def main(opt):

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    start_epoch = 1
    last_path = os.path.join(opt.save_folder, 'last.pth')
    if os.path.isfile(last_path):
        print(f'=> Resuming from {last_path}')
        ckpt = torch.load(last_path, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)      
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except Exception as e:
            print(f'! Optimizer state not loaded: {e}')
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        print(f'=> Continue training at epoch {start_epoch}')

    model = model.cuda()
    
    # training routine
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss_vicl_performance, loss_label_matching, loss_loaded_avg = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.add_scalar('loss_vicl_performance', loss_vicl_performance, epoch)
        logger.add_scalar('loss_label_matching', loss_label_matching, epoch)
        logger.add_scalar('loss_loaded_avg', loss_loaded_avg, epoch)
        logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # logger.log_value('loss', loss, epoch)
        # logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)
        
        if epoch % 10 == 0:
            save_file = os.path.join(
                opt.save_folder, 'last.pth')
            save_model(model, optimizer, opt, epoch, save_file)
    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
    logger.close()


if __name__ == '__main__':
    opt = parse_option()
    print(opt)
    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(opt)





    
