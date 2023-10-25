# -*- coding:utf-8 -*-
from __future__ import absolute_import, division
import sys
import time
import datetime
import argparse
import os
import os.path as osp
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# from util.heatmaps import draw_heatmap
# from pyheatmap.heatmap import HeatMap

import copy
import models
from util.losses import CrossEntropyLoss, DeepSupervision, CrossEntropyLabelSmooth, TripletLossAlignedReID
from util import data_manager
from util import transforms as T
from util.dataset_loader import ImageDataset
from util.utils import AverageMeter, Logger, save_checkpoint
from util.eval_metrics import evaluate
from util.optimizers import init_optim
from util.samplers import RandomIdentitySampler
from IPython import embed
from numpy import arange

parser = argparse.ArgumentParser(description='Train AlignedReID with cross entropy loss and triplet hard loss')
# Datasets
parser.add_argument('--root', type=str, default='data', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=256,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0, help="split index")


# CUHK03-specific setting
parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use-metric-cuhk03', action='store_true',
                    help="whether to use cuhk03-metric (default: False)")
# Optimization options
parser.add_argument('--labelsmooth', action='store_true', help="label smooth")
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max_epoch', default=3, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=32, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=150, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
# triplet hard loss
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())

# Miscs
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')

parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save_dir', type=str, default='log')
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--reranking',action= 'store_true', help= 'result re_ranking')

parser.add_argument('--test_distance',type = str, default='global', help= 'test distance type')
parser.add_argument('--unaligned',action= 'store_true', help= 'test local feature with unalignment')

args = parser.parse_args()

np.seterr(divide='ignore', invalid='ignore')

Loss, Loss_every_epoch = [], []
mAP_list, rank1_list, rank5_list, rank10_list = [], [], [], []
best_mAP_list, best_rank1_list, best_rank5_list, best_rank10_list = [], [], [], []

label_0 = 'Loss(epoch)-Curve on {}'.format(args.dataset)
label_1 = 'Loss(iter)-Curve on {}'.format(args.dataset)
labels_1 = ['losses']

label_2 = 'mAP-Curve on {}'.format(args.dataset)
labels_2 = ['mAP']

label_3 = 'rank-Curve on {}'.format(args.dataset)
labels_3 = ['rank1', 'rank5', 'rank10']

label_4 = 'mAP(best)-Curve on {}'.format(args.dataset)
labels_4 = ['best_mAP']

label_5 = 'rank(best)-Curve on {}'.format(args.dataset)
labels_5 = ['best_rank1', 'best_rank5', 'best_rank10']

# fig =plt.figure()
fig0, ax0 = plt.subplots()
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()

def main():
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    pin_memory = True if use_gpu else False

    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(osp.join(args.save_dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')  

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_img_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id,
        cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
    )


    # data augmentation
    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'softmax','metric'}, aligned =True, use_gpu=use_gpu)
    model = model.cuda()
    # print('model_aligned_resnet50', model)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    if args.labelsmooth:
        criterion_class = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    else:
        criterion_class = CrossEntropyLoss(use_gpu=use_gpu)
    criterion_metric = TripletLossAlignedReID(margin=args.margin)
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)

    if args.stepsize > 0:
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
        #lambda1 = lambda epoch: epoch // 30
        #lambda2 = lambda epoch: 0.95 ** epoch
        #scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    start_epoch = args.start_epoch
    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

        optimizer.load_state_dict(checkpoint['optimizer'])

    if use_gpu:
        # 多GPU optimizer：optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        test(model, queryloader, galleryloader, use_gpu)
        return 0

    start_time = time.time()
    train_time = 0

    best_mAP = -np.inf
    best_rank1, best_rank5, best_rank10 = -np.inf, -np.inf, -np.inf
    best_mAP_epoch = 0
    best_rank1_epoch, best_rank5_epoch, best_rank10_epoch = 0, 0, 0

    # heatmap_outdir_root = 'heatmap/{}/{}'.format(args.dataset, 'outputs_features_distmat')
    # if not osp.exists(heatmap_outdir_root):
    #     os.makedirs(heatmap_outdir_root)

    # heatmap_outdir_train = osp.join(heatmap_outdir_root, 'outputs_features_distmat_train')
    # if not osp.exists(heatmap_outdir_train):
    #     os.makedirs(heatmap_outdir_train)
    # heatmap_outdir_query_gallery = osp.join(heatmap_outdir_root, 'outputs_features_distmat_query_gallary')
    # if not osp.exists(heatmap_outdir_query_gallery):
    #     os.makedirs(heatmap_outdir_query_gallery)

    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):

        mAP_txt = open(osp.join(args.save_dir, 'mAP({}).txt'.format(args.dataset)), 'a+')
        rank_txt = open(osp.join(args.save_dir, 'rank({}).txt'.format(args.dataset)), 'a+')
        best_mAP_txt = open(osp.join(args.save_dir, 'best_mAP({}).txt'.format(args.dataset)), 'a+')
        best_rank_txt = open(osp.join(args.save_dir, 'best_rank({}).txt'.format(args.dataset)), 'a+')

        start_train_time = time.time()
        train_result = train(epoch, model, criterion_class, criterion_metric, optimizer, trainloader, use_gpu)
        # train(epoch, model, criterion_class, criterion_metric, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)

        if args.stepsize > 0: scheduler.step(train_result['loss'])

        outputs, features, local_features = train_result['outputs'], train_result['features'], train_result['local_features']
        
        # dpi =100
        # w_in_inch, h_in_inch = float(args.width)/dpi, float(args.height)/dpi
        # gc = plt.gcf()
        # gc.set_size_inches(w_in_inch, h_in_inch)
        # draw_heatmap(outputs.detach().cpu().numpy())
        # plt.title('outputs')
        # plt.savefig('{}/{}.jpg'.format(heatmap_outdir_train, epoch))

        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")
            test_result = test(model, queryloader, galleryloader, use_gpu)

            is_best = test_result['rank1'] > best_rank1
            if is_best:
                best_rank1 = test_result['rank1']
                best_rank1_epoch = epoch + 1

            is_best = test_result['rank5'] > best_rank5
            if is_best:
                best_rank5 = test_result['rank5']
                best_rank5_epoch = epoch + 1

            is_best = test_result['rank10'] > best_rank10
            if is_best:
                best_rank10 = test_result['rank10']
                best_rank10_epoch = epoch + 1

            is_best = test_result['mAP'] > best_mAP
            if is_best:
                best_mAP = test_result['mAP']
                best_mAP_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

        # distmat = test_result['distmat']
        # draw_heatmap(distmat)
        # plt.title('distmat')
        # plt.savefig('{}/{}.jpg'.format(heatmap_outdir_query_gallery, epoch))

        # distmat_list = distmat.tolist()
        # print('distmat_list_type', distmat_list)
        # hm = HeatMap(distmat_list[0])
        # hm.clickmap(save_as='{}/hit_distmat.png'.format(heatmap_outdir_query_gallery))
        # hm.heatmap(save_as='{}/eat_distmat.png'.format(heatmap_outdir_query_gallery))

        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint({
                    'state_dict': state_dict,
                    'test_result': test_result,
                    'epoch': epoch+1,
                    'optimizer': optimizer.state_dict(),
                    }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

        mAP_list.append(test_result['mAP'])
        rank1_list.append(test_result['rank1'])
        rank5_list.append(test_result['rank5'])
        rank10_list.append(test_result['rank10'])

        best_mAP_list.append(best_mAP)
        best_rank1_list.append(best_rank1)
        best_rank5_list.append(best_rank5)
        best_rank10_list.append(best_rank10)

        x1 = np.linspace(1, len(Loss), len(Loss))
        ax1.set_title(label_1)
        # ax1.set_xticklabels(x1)
        # ax1.set_yticklabels(Loss)
        ax1.set_xlabel('iterations')
        ax1.set_ylabel('losses')
        ax1.grid(True, ls= '--')
        ax1.plot(x1, Loss, 'r-', mec='k', lw=1)
        fig1.savefig('{}/train_loss_iter.jpg'.format(args.save_dir))

        # x2 = np.arange(len(mAP_list))
        x2 = np.linspace(1, len(mAP_list), len(mAP_list))
        ax2.set_title(label_2)
        # ax2.set_xticklabels(x2)
        # ax2.set_yticklabels(mAP_list)
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('mAP')
        ax2.grid(True, ls= '--')
        ax2.plot(x2, mAP_list, 'b-', mec='k', lw=1)
        fig2.savefig('{}/test_mAP_epoch{}.jpg'.format(args.save_dir, args.max_epoch))

        x3 = np.linspace(1, len(rank1_list), len(rank1_list))
        ax3.set_title(label_3)
        # ax3.set_xticklabels(x3)
        # ax3.set_yticklabels(rank1_list)
        ax3.set_xlabel('epochs')
        ax3.set_ylabel('rank')
        ax3.grid(True, ls= '--')
        ax3.plot(x3, rank1_list, 'r--', mec='k', lw=1)
        ax3.plot(x3, rank5_list, 'g-', mec='k', lw=1)
        ax3.plot(x3, rank10_list, 'b:', mec='k', lw=1)
        fig3.savefig('{}/test_rank_epoch{}.jpg'.format(args.save_dir, args.max_epoch))

        x4 = np.linspace(1, len(best_mAP_list), len(best_mAP_list))
        ax4.set_title(label_4)
        # ax4.set_xticklabels(x4)
        # ax4.set_yticklabels(best_mAP_list)
        ax4.set_xlabel('epochs')
        ax4.set_ylabel('best mAP')
        ax4.grid(True, ls= '--')
        ax4.plot(x4, best_mAP_list, 'b-', mec='k', lw=1)
        fig4.savefig('{}/test_mAP(best)_epoch{}.jpg'.format(args.save_dir, args.max_epoch)) 

        x5 = np.linspace(1, len(best_rank1_list), len(best_rank1_list))
        ax5.set_title(label_5)
        # ax5.set_xticklabels(x5)
        # ax5.set_yticklabels(best_rank1_list)
        ax5.set_xlabel('epochs')
        ax5.set_ylabel('rank')
        ax5.grid(True, ls= '--')
        ax5.plot(x5, best_rank1_list, 'r--', mec='k', lw=1)
        ax5.plot(x5, best_rank5_list, 'g-', mec='k', lw=1)
        ax5.plot(x5, best_rank10_list, 'b:', mec='k', lw=1)
        fig5.savefig('{}/test_rank(best)_epoch{}.jpg'.format(args.save_dir, args.max_epoch))


        mAP_value = 'Epoch ' + str(epoch+1) + ' mAP ' + str(test_result['mAP']) + '\n'
        rank_value = 'Epoch ' + str(epoch+1) + ' rank1 ' + str(test_result['rank1']) + '\n' + \
                    'Epoch ' + str(epoch+1) + ' rank5 ' + str(test_result['rank5']) + '\n' + \
                    'Epoch ' + str(epoch+1) + ' rank10 ' + str(test_result['rank10']) + '\n' + \
                    '----------------' + '\n' 
        mAP_txt.write(mAP_value)
        rank_txt.write(rank_value)

        best_mAP_value = 'Epoch ' + str(epoch+1) + ' best_mAP ' + str(best_mAP) + '\n'
        best_rank_value = 'Epoch ' + str(epoch+1) + ' best_rank1 ' + str(best_rank1) + '\n' + \
                    'Epoch ' + str(epoch+1) + ' best_rank5 ' + str(best_rank5) + '\n' + \
                    'Epoch ' + str(epoch+1) + ' best_rank10 ' + str(best_rank10) + '\n' + \
                    '----------------' + '\n' 
        best_mAP_txt.write(best_mAP_value)
        best_rank_txt.write(best_rank_value)


    mAP_txt.close()
    rank_txt.close()
    best_mAP_txt.close()
    best_rank_txt.close()

    if args.max_epoch % args.save_freq != 0:
        save_checkpoint({
        'state_dict': state_dict,
        'test_result': test_result,
        'epoch': args.max_epoch,
        'optimizer': optimizer.state_dict(),
        }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))
 
    # x1 = np.arange(len(Loss))
    x0 = np.linspace(1, args.max_epoch, args.max_epoch)
    ax0.set_title(label_1)
    # ax2.set_xticks(range(len(x0)))
    # ax0.set_xticklabels(x0)  
    # ax0.set_yticklabels(Loss_every_epoch)
    ax0.set_xlabel('epochs')
    ax0.set_ylabel('losses')
    ax0.grid(True, ls= '--')
    ax0.plot(x0, Loss_every_epoch, 'r-', mec='k', label=labels_1[0], lw=1)
    ax0.legend(loc = 'best', fontsize=8)
    fig0.savefig('{}/train_loss_epoch{}.jpg'.format(args.save_dir, args.max_epoch))

    ax1.plot(x1, Loss, 'r-', mec='k', label=labels_1[0], lw=1)
    # ax1.legend((labels_1[0]), loc = 'best', fontsize=8)
    ax1.legend(loc = 'best', fontsize=8)
    fig1.savefig('{}/train_loss_iter.jpg'.format(args.save_dir))

    ax2.plot(x2, mAP_list, 'b-', mec='k', label=labels_2[0], lw=1)
    # ax2.legend((labels_2[0]), loc = 'best', fontsize=8)
    ax2.legend(loc = 'best', fontsize=8)
    fig2.savefig('{}/test_mAP_epoch{}.jpg'.format(args.save_dir, args.max_epoch))

    # ax3.plot(x3, rank1_list, 'r-', mec='k', label=labels_3[0], lw=1)
    # ax3.plot(x3, rank5_list, 'g-', mec='k', label=labels_3[1], lw=1)
    # ax3.plot(x3, rank10_list, 'b-', mec='k', label=labels_3[2], lw=1)
    ax3.legend((labels_3[0], labels_3[1], labels_3[2]), loc = 'best', fontsize=8)
    # ax3.legend(loc = 'best', fontsize=8)
    fig3.savefig('{}/test_rank_epoch{}.jpg'.format(args.save_dir, args.max_epoch))

    ax4.plot(x4, best_mAP_list, 'b-', mec='k', label=labels_4[0], lw=1)
    ax4.legend(loc = 'best', fontsize=8)
    fig4.savefig('{}/test_mAP(best)_epoch{}.jpg'.format(args.save_dir, args.max_epoch)) 

    # ax5.plot(x5, best_rank1_list, 'r-', mec='k', label=labels_5[0], lw=1)
    # ax5.plot(x5, best_rank5_list, 'g-', mec='k', label=labels_5[1], lw=1)
    # ax5.plot(x5, best_rank10_list, 'b-', mec='k', label=labels_5[2], lw=1)
    # ax5.legend(loc = 'best', fontsize=8)
    ax5.legend((labels_5[0], labels_5[1], labels_5[2]), loc = 'best', fontsize=8)
    fig5.savefig('{}/test_rank(best)_epoch{}.jpg'.format(args.save_dir, args.max_epoch))

    print('==> Best mAP {:.1%}, achieved at epoch {}'.format(best_mAP, best_mAP_epoch))
    print("==> Best Rank1 {:.1%}, achieved at epoch {}".format(best_rank1, best_rank1_epoch))
    print("==> Best Rank5 {:.1%}, achieved at epoch {}".format(best_rank5, best_rank5_epoch))
    print("==> Best Rank10 {:.1%}, achieved at epoch {}".format(best_rank10, best_rank10_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

def train( epoch, model, criterion_class, criterion_metric, optimizer, trainloader, use_gpu):
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    xent_losses = AverageMeter()
    global_losses = AverageMeter()
    local_losses = AverageMeter()

    end = time.time()

    loss_txt = open(osp.join(args.save_dir, 'loss({}).txt'.format(args.dataset)), 'a+')

    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
        
        # measure data loading time
        data_time.update(time.time() - end)
        outputs, features, local_features = model(imgs)

        if args.htri_only:
            if isinstance(features, tuple):
                global_loss, local_loss = DeepSupervision(criterion_metric, features, pids, local_features)
            else:
                global_loss, local_loss = criterion_metric(features, pids, local_features)
        else:
            if isinstance(outputs, tuple):
                xent_loss = DeepSupervision(criterion_class, outputs, pids)
            else:
                xent_loss = criterion_class(outputs, pids)

            if isinstance(features, tuple):
                global_loss, local_loss = DeepSupervision(criterion_metric, features, pids, local_features)
            else:
                global_loss, local_loss = criterion_metric(features, pids, local_features)
                
        loss = xent_loss + global_loss + local_loss  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), pids.size(0))
        xent_losses.update(xent_loss.item(), pids.size(0))
        global_losses.update(global_loss.item(), pids.size(0))
        local_losses.update(local_loss.item(), pids.size(0))

        Loss.append(losses.val)

        if (batch_idx+1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'CLoss {xent_loss.val:.4f} ({xent_loss.avg:.4f})\t'
                  'GLoss {global_loss.val:.4f} ({global_loss.avg:.4f})\t'
                  'LLoss {local_loss.val:.4f} ({local_loss.avg:.4f})\t'.format(
                   epoch+1, batch_idx+1, len(trainloader), batch_time=batch_time,data_time=data_time,
                   loss=losses,xent_loss=xent_losses, global_loss=global_losses, local_loss = local_losses))
 
    # a = reduce(lambda x,y: x+y, Loss)/len(Loss)
    # l.append(a)
    Loss_every_epoch.append(losses.avg)

    loss_value = 'Epoch ' + str(epoch+1) + ' loss ' + str(losses.avg) + '\n'
    loss_txt.write(loss_value)
    loss_txt.close()

    return { 'batch_idx':batch_idx, 'loss':losses.val, 'outputs':outputs, 
            'features':features, 'local_features':local_features, 'pids': pids}

def test( model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids, lqf = [], [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features, local_features = model(imgs)
            print('features', features.shape)
            print('local_features', local_features.shape)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            local_features = local_features.data.cpu()
            qf.append(features)
            lqf.append(local_features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        lqf = torch.cat(lqf,0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids, lgf = [], [], [], []
        end = time.time()
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features, local_features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            local_features = local_features.data.cpu()
            gf.append(features)
            lgf.append(local_features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        lgf = torch.cat(lgf,0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)


        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))
    # feature normlization
    # torch.norm(input, p=2) → float 返回输入张量input的p范数
    qf = 1. * qf / (torch.norm(qf, 2, dim = -1, keepdim=True).expand_as(qf) + 1e-12)
    gf = 1. * gf / (torch.norm(gf, 2, dim = -1, keepdim=True).expand_as(gf) + 1e-12)
    m, n = qf.size(0), gf.size(0)

    # distmat是矩阵,其中(i,j)元素表示第i个点和第j个点之间的距离
    # torch.pow:求幂, keepdim=True表示保持原有的空间形状
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    if args.dataset != 'veri':
        q_camids = np.zeros(m).astype(np.int32)
        g_camids = np.ones(n).astype(np.int32)

    if not args.test_distance== 'global':
        print("Only using global branch")
        from util.distance import low_memory_local_dist
        lqf = lqf.permute(0,2,1)
        lgf = lgf.permute(0,2,1)
        local_distmat = low_memory_local_dist(lqf.numpy(),lgf.numpy(),aligned= not args.unaligned)
        if args.test_distance== 'local':
            print("Only using local branch")
            distmat = local_distmat
        if args.test_distance == 'global_local':
            print("Using global and local branches")
            distmat = local_distmat+distmat
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    # print("Results ----------")
    # print("mAP: {:.1%}".format(mAP))
    # print("CMC curve")
    # for r in ranks:
    #     print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    # print("------------------")

    if args.reranking:
        from util.re_ranking import re_ranking
        if args.test_distance == 'global':
            print("Only using global branch for reranking")
            distmat = re_ranking(qf,gf,k1=20, k2=6, lambda_value=0.3)
        else:
            local_qq_distmat = low_memory_local_dist(lqf.numpy(), lqf.numpy(),aligned= not args.unaligned)
            local_gg_distmat = low_memory_local_dist(lgf.numpy(), lgf.numpy(),aligned= not args.unaligned)
            local_dist = np.concatenate(
                [np.concatenate([local_qq_distmat, local_distmat], axis=1),
                 np.concatenate([local_distmat.T, local_gg_distmat], axis=1)],
                axis=0)
            if args.test_distance == 'local':
                print("Only using local branch for reranking")
                distmat = re_ranking(qf,gf,k1=20,k2=6,lambda_value=0.3,local_distmat=local_dist,only_local=True)
            elif args.test_distance == 'global_local':
                print("Using global and local branches for reranking")
                distmat = re_ranking(qf,gf,k1=20,k2=6,lambda_value=0.3,local_distmat=local_dist,only_local=False)
                
        print('distmat', np.shape(distmat))
        print('distmat_type', type(distmat))

        print("Computing CMC and mAP for re_ranking")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)
        
        print("Results ----------")
        print("mAP(RK): {:.1%}".format(mAP))
        print("CMC curve(RK)")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")

    return {'rank1': cmc[0], 'rank5': cmc[4], 'rank10': cmc[9], 'mAP': mAP, 'distmat': distmat}

if __name__ == '__main__':
    main()
