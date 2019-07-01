#coding=utf-8
from __future__ import absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
from copy import deepcopy
import PIL.Image as Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
import copy
import shutil
import models
import tabulate
from util.utils import adjust_learning_rate, moving_average, bn_update
from util.losses import CrossEntropyLoss, DeepSupervision, CrossEntropyLabelSmooth, TripletLossAlignedReID
from util import data_manager
from util import transforms as T
from util.dataset_loader import ImageDataset
from util.utils import Logger
from util.utils import AverageMeter, Logger, save_checkpoint
from util.eval_metrics import evaluate
from util.optimizers import init_optim
from util.samplers import RandomIdentitySampler
from IPython import embed

# argparse :命令行解析. 创建解析器对象ArgumentParser，可以添加参数。description：描述程序
parser = argparse.ArgumentParser(description='Train AlignedReID with cross entropy loss and triplet hard loss')
# Datasets, add_argument()方法，用来指定程序需要接受的命令参数
parser.add_argument('--root', type=str, default='data', help="root path to data directory")  
parser.add_argument('-d', '--dataset', type=str, default='boxcars116k',
                    choices=data_manager.get_names())                
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=256,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0, help="split index")

# parser.add_argument('--use_metric_noncamid', action='store_true',
#                     help="whether to use noncamid_metric (default: False)")

# CUHK03-specific setting
parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use-metric-cuhk03', action='store_true',
                    help="whether to use cuhk03-metric (default: False)")
# Optimization options 优化
parser.add_argument('--labelsmooth', action='store_true', help="label smooth")      
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--swa_epoch', default=161, type=int, help="optimization algorithm (see optimizers.py)")
# 迭代终止时的那个 epoch，不管从第几个 epoch 开始，都计算到这个 epoch 就停止
parser.add_argument('--max_epoch', default=300, type=int, help="maximum epochs to run")  # num-epoch --> max-epoch    
parser.add_argument('--start-epoch', default=0, type=int, help="manual epoch number (useful on restarts)")   # 接着上次训练时从哪个 epoch 开始
parser.add_argument('--train-batch', default=32, type=int, help="train batch size")
parser.add_argument('--test-batch', default=32, type=int, help="test batch size")

# lr:网络的基础学习速率,一般设一个很小的值,然后根据迭代到不同次数,对学习速率做相应的变化.lr过大不会收敛,过小收敛过慢
parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=150, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")  # 每150次epoch衰减学习率的大小:：乘以gamma
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")                                         # 学习率变化的比率
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")                  # 权值衰量,用于防止过拟合               

parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
parser.add_argument('--swa_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR', help='SWA LR (default: 0.05)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')

# triplet hard loss
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")      # 三元组损失误差的达标要求
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")                # 每一个 identity 的例子数
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")    # 是否只用三元组损失中的难三元组损失进行训练
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())    # 使用哪个神经网络模型
# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")

# resume:训练好的模型所在的路径，测试时需要从该路径下导入模型。如果想接着上次的模型继续训练，可以从这里导入上次的模型
# parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--resume', type=str, default=None, metavar='checkpoint', help='checkpoint to resume training from (default: None)')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")  # 使用该选项后，程序只进行测试，不再训练
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save_dir', type=str, default='log')          # 测试时输出的信息将会保存在这个路径下面
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--reranking', action= 'store_true', help= 'result re_ranking')

parser.add_argument('--test_distance',type = str, default='global', help= 'test distance type')
parser.add_argument('--unaligned',action= 'store_true', help= 'test local feature with unalignment')

# parser.add_argument('--stages', type=str, metavar='STAGE DEPTH',
#                     help='per layer depth')
# parser.add_argument('--bottleneck', default=4, type=int, metavar='B',
#                     help='bottleneck (default: 4)')
# parser.add_argument('--group-1x1', type=int, metavar='G', default=4,
#                     help='1x1 group convolution (default: 4)')
# parser.add_argument('--group-3x3', type=int, metavar='G', default=4,
#                     help='3x3 group convolution (default: 4)')
# parser.add_argument('--condense-factor', type=int, metavar='C', default=4,
#                     help='condense factor (default: 4)')
# parser.add_argument('--growth', type=str, metavar='GROWTH RATE',
#                     help='per layer growth')
# parser.add_argument('--reduction', default=0.5, type=float, metavar='R',
#                     help='transition reduction (default: 0.5)')
# parser.add_argument('--dropout-rate', default=0, type=float,
#                     help='drop out (default: 0)')
# parser.add_argument('--group-lasso-lambda', default=0., type=float, metavar='LASSO',
#                     help='group lasso loss weight (default: 0)')


args = parser.parse_args()
np.seterr(divide='ignore',invalid='ignore')

Loss_every_epoch, Loss = [], []

init_mAP_list, swa_mAP_list, best_mAP_list = [], [], []

init_rank1_list, swa_rank1_list, best_rank1_list = [], [], []
init_rank5_list, swa_rank5_list, best_rank5_list = [], [], []
init_rank10_list, swa_rank10_list, best_rank10_list = [], [], []


label_0 = 'Loss(epoch)-Curve on {}'.format(args.dataset)
label_1 = 'Loss(iter)-Curve on {}'.format(args.dataset)
labels_1 = ['losses']

label_2 = 'mAP(init)-Curve on {}'.format(args.dataset)
labels_2 = ['init_mAP']
label_3 = 'rank(init)-Curve on {}'.format(args.dataset)
labels_3 = ['init_rank1', 'init_rank5', 'init_rank10']

label_4 = 'mAP(swa)-Curve on {}'.format(args.dataset)
labels_4 = ['swa_mAP']
label_5 = 'rank(swa)-Curve on {}'.format(args.dataset)
labels_5 = ['swa_rank1', 'swa_rank5', 'swa_rank10']

label_6 = 'mAP(best)-Curve on {}'.format(args.dataset)
labels_6 = ['best_mAP']
label_7 = 'rank(best)-Curve on {}'.format(args.dataset)
labels_7 = ['best_rank1', 'best_rank5', 'best_rank10']

# fig =plt.figure()
fig0, ax0 = plt.subplots()
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()
fig6, ax6 = plt.subplots()
fig7, ax7 = plt.subplots()

# 默认应用gpu
def main():
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    pin_memory = True if use_gpu else False

    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))  # 输出日志
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))              # .format 格式化

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        # 为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_img_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id,
        cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
    )
    # data augmentation 数据增强
    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        #T.RandomResizedCrop(256, scale=(0.1, 1), interpolation=Image.BILINEAR),
        # T.RandomRotation(1, resample=False, expand=False, center=None),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 导入训练集和测试集
    # dataset.train: (img_path, pid, camid)
    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        # sampler:定义一个方法来绘制样本数据，如果定义该方法，则不能使用shuffle(是否打乱数据，默认为False)
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        # num_workers:数据分为几批处理（对于大数据）
        batch_size=args.train_batch, num_workers=args.workers,
        # pin_memory: 针对不同类型的batch进行处理。比如为Map或者Squence等类型，需要处理为tensor类型。
        # drop_last:用于处理最后一个batch的数据。因为最后一个可能不能够被整除，如果设置为True，则舍弃最后一个，为False则保留最后一个，但是最后一个可能很小。
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

    # 训练/导入模型
    print("Initializing model: {}".format(args.arch))
    # 初始化的网络模型
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'softmax', 'metric'}, aligned=True, use_gpu=use_gpu)
    model.cuda()
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    if args.swa:
        print('SWA training:')
        swa_model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'softmax', 'metric'}, aligned=True, use_gpu=use_gpu)
        swa_model.cuda()
        swa_n = 0
    else:
        print('Init training:')

    def schedule(epoch):
        t = (epoch) / (args.swa_start if args.swa else args.max_epoch)
        lr_ratio = args.swa_lr / args.lr if args.swa else 0.01
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return args.lr * factor

    if args.labelsmooth:
        criterion_class = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    else:
        # We chose cross entropy loss as the classification loss.
        criterion_class = CrossEntropyLoss(use_gpu=use_gpu)   # 实例化的交叉熵损失函数：分类损失。交叉熵损失作为分类损失
    # 实例化的三元组损失函数：度量损失。把三元组损失作为度量损失
    criterion_metric = TripletLossAlignedReID(margin=args.margin)

    # 新建一个优化器，指定优化方法、要调整的 model 参数、学习率、权重衰减
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)    # optim: 最优化

    # if args.stepsize > 0:
    #     scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

        # 以余弦函数为周期，并在每个周期最大值时重新设置学习率。以初始学习率为最大学习率，以 2∗Tmax 2*Tmax2∗Tmax 为周期，在一个周期内先下降，后上升。
        # T_max(int)- 一次学习率周期的迭代次数，即 T_max 个 epoch 之后重新设置学习率。
        # eta_min(float)- 最小学习率，即在一个周期中，学习率最小会下降到 eta_min，默认值为 0。
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=4e-08)


        # 自定义调整学习率 LambdaLR
        #为不同参数组设定不同学习率调整策略。调整规则为，lr=base_lr∗lmbda(self.last_epoch)
        #fine-tune 中十分有用，我们不仅可为不同的层设定不同的学习率，还可以为其设定不同的学习率调整策略。
        #lambda1 = lambda epoch: epoch // 30
        #lambda2 = lambda epoch: 0.95 ** epoch
        #scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

        # """
        # 自适应调整学习率 ReduceLROnPlateau
        # 当某指标不再变化（下降或升高），调整学习率，这是非常实用的学习率调整策略。
        # 例如，当验证集的 loss 不再下降时，进行学习率调整；或者监测验证集的 accuracy，当
        # accuracy 不再上升时，则调整学习率。

        # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, 
        # verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

        # 参数：
        # mode(str)- 模式选择，有 min 和 max 两种模式， min 表示当指标不再降低(如监测loss)， max 表示当指标不再升高(如监测 accuracy)。
        # factor(float)- 学习率调整倍数(等同于其它方法的 gamma)，即学习率更新为 lr = lr * factor
        # patience(int)- 忍受该指标多少个 step 不变化，当忍无可忍时，调整学习率。
        # verbose(bool)- 是否打印学习率信息， print(‘Epoch {:5d}: reducing learning rate’
        # ’ of group {} to {:.4e}.’.format(epoch, i, new_lr))
        # threshold_mode(str)- 选择判断指标是否达最优的模式，有两种模式， rel 和 abs。
        # 当 threshold_mode == rel，并且 mode == max 时， dynamic_threshold = best * ( 1 +threshold )；
        # 当 threshold_mode == rel，并且 mode == min 时， dynamic_threshold = best * ( 1 -threshold )；
        # 当 threshold_mode == abs，并且 mode== max 时， dynamic_threshold = best + threshold ；
        # 当 threshold_mode == rel，并且 mode == max 时， dynamic_threshold = best - threshold
        # threshold(float)- 配合 threshold_mode 使用。
        # cooldown(int)- “冷却时间“，当调整学习率之后，让学习率调整策略冷静一下，让模型再训
        # 练一段时间，再重启监测模式。
        # min_lr(float or list)- 学习率下限，可为 float，或者 list，当有多个参数组时，可用 list 进行设置。
        # eps(float)- 学习率衰减的最小值，当学习率变化小于 eps 时，则不调整学习率。
        # 原文：https://blog.csdn.net/shanglianlm/article/details/85143614 
        # """
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True,'min')

        # Assuming optimizer has two groups.
        # lambda1 = lambda epoch: epoch // 30
        # lambda2 = lambda epoch: 0.95 ** epoch
        # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        # Assuming optimizer uses lr = 0.05 for all groups
        # lr = 0.05   if epoch < 30, lr = 0.005  if 30 <= epoch < 80, lr = 0.0005   if epoch >= 80
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300], gamma=args.gamma, last_epoch=-1)

        # lr_scheduler.ExponentialLR(optimizer, args.gamma, last_epoch=-1)

        # # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)  #　T_max：最大迭代次数
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.train_batch * args.max_epoch, eta_min=0, last_epoch=-1)

   
    start_epoch = args.start_epoch
    # 导入已有的模型
    # if args.resume:
        # print("Loading checkpoint from '{}'".format(args.resume))
    if args.resume is not None:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        state_dict = checkpoint['state_dict']
        # best_init_rank1 = checkpoint['best_init_rank1']
        start_epoch = checkpoint['epoch']

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        if args.swa:
            # best_swa_rank1 = checkpoint['best_swa_rank1']
            swa_state_dict = checkpoint['swa_state_dict']
            if swa_state_dict is not None:
                swa_model.load_state_dict(swa_state_dict)
            swa_n_ckpt = checkpoint['swa_n']
            if swa_n_ckpt is not None:
                swa_n = swa_n_ckpt
    
    if args.swa:
        swa_test_result = {'rank1': None, 'rank5': None, 'rank10': None, 'mAP': None}

    if use_gpu:
        model = nn.DataParallel(model).cuda()   # 如果有多个GPU的话
        if args.swa:
            swa_model = nn.DataParallel(swa_model).cuda()

    if args.evaluate:
        print("Evaluate only")
        test(model, queryloader, galleryloader, use_gpu)
        return 0

    start_time = time.time()
    train_time = 0

    # inf -> ∞
    best_init_mAP, best_swa_mAP, best_mAP = -np.inf, -np.inf, -np.inf
    best_init_rank1, best_init_rank5, best_init_rank10 = -np.inf, -np.inf, -np.inf        
    best_swa_rank1, best_swa_rank5, best_swa_rank10 = -np.inf, -np.inf, -np.inf
    best_rank1, best_rank5, best_rank10 = -np.inf, -np.inf, -np.inf
    
    best_init_mAP_epoch, best_swa_mAP_epoch, best_mAP_epoch = 0, 0, 0
    best_init_rank1_epoch, best_init_rank5_epoch, best_init_rank10_epoch = 0, 0, 0      
    best_swa_rank1_epoch, best_swa_rank5_epoch, best_swa_rank10_epoch = 0, 0, 0
    best_rank1_epoch, best_rank5_epoch, best_rank10_epoch = 0, 0, 0 
    init_epochs, swa_epochs, epochs = [], [], []

    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):

        init_mAP_txt = open(osp.join(args.save_dir, 'init_mAP({}).txt'.format(args.dataset)), 'a+')
        swa_mAP_txt = open(osp.join(args.save_dir, 'swa_mAP({}).txt'.format(args.dataset)), 'a+')
        best_mAP_txt = open(osp.join(args.save_dir, 'best_mAP({}).txt'.format(args.dataset)), 'a+')

        init_rank_txt = open(osp.join(args.save_dir, 'init_rank({}).txt'.format(args.dataset)), 'a+')
        swa_rank_txt = open(osp.join(args.save_dir, 'swa_rank({}).txt'.format(args.dataset)), 'a+')
        best_rank_txt = open(osp.join(args.save_dir, 'best_rank({}).txt'.format(args.dataset)), 'a+')

        epochs.append(epoch) 

        # 从这个函数进入到训练过程，输入当前 epoch、网络模型、分类损失、度量损失、优化器、训练集
        start_train_time = time.time()
        # train(epoch, model, criterion_class, criterion_metric, optimizer, trainloader, use_gpu)
        train_result = train(epoch, model, criterion_class, criterion_metric, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)

        # Assuming optimizer has two groups.
        # lambda1 = lambda epoch: epoch // 30
        # lambda2 = lambda epoch: 0.95 ** epoch
        # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2], last_epoch=-1)
        
        lr = schedule(epoch)
        print('lr', lr)
        adjust_learning_rate(optimizer, lr)

        # if args.stepsize > 0: scheduler.step()
        # if args.stepsize > 0 and epoch <= 200: scheduler.step(train_result['loss'])
        # if args.stepsize > 0: scheduler.step(train_result['loss'])

        # if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch:
        # epoch % args.eval_freq == args.eval_freq - 1 : 假设eval_freq是25 ，则当(epoch+1)是25的倍数是，保存一次模型
        # if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.max_epoch - 1:
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.max_epoch - 1:
            print("==> Test")
            test_result = test(model, queryloader, galleryloader, use_gpu, Swa=False)
        else:
            test_result = {'rank1': None, 'rank5': None, 'rank10': None, 'mAP': None}

        if test_result['rank1'] != None:
            is_best = test_result['rank1'] > best_init_rank1
            if is_best:
                best_init_rank1 = test_result['rank1']
                best_init_rank1_epoch = epoch + 1

            is_best = test_result['mAP'] > best_init_mAP
            if is_best:
                best_init_mAP = test_result['mAP']
                best_init_mAP_epoch = epoch + 1

            is_best = test_result['rank5'] > best_init_rank5
            if is_best:
                best_init_rank5 = test_result['rank5']
                best_init_rank5_epoch = epoch + 1

            is_best = test_result['rank10'] > best_init_rank10
            if is_best:
                best_init_rank10 = test_result['rank10']
                best_init_rank10_epoch = epoch + 1

            init_mAP_list.append(test_result['mAP'])
            init_rank1_list.append(test_result['rank1']) 
            init_rank5_list.append(test_result['rank5']) 
            init_rank10_list.append(test_result['rank10']) 
            init_epochs.append(epoch)

        if args.swa and (epoch + 1) >= args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
            moving_average(swa_model, model, 1.0 / (swa_n + 1))
            swa_n += 1
            print('swa_n', swa_n)

            # if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.max_epoch - 1:
            if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.max_epoch - 1:
                bn_update(trainloader, swa_model)
                print("==> Swa Test")
                swa_test_result = test(swa_model, queryloader, galleryloader, use_gpu, Swa=True)
            else:
                swa_test_result = {'rank1': None, 'rank5': None, 'rank10': None, 'mAP': None}

            if swa_test_result['rank1'] != None:
                is_best = swa_test_result['rank1'] > best_swa_rank1 
                if is_best:
                    best_swa_rank1 = swa_test_result['rank1']
                    best_swa_rank1_epoch = epoch + 1

                is_best = swa_test_result['mAP'] > best_swa_mAP 
                if is_best:
                    best_swa_mAP = swa_test_result['mAP']
                    best_swa_mAP_epoch = epoch + 1

                is_best = swa_test_result['rank5'] > best_swa_rank5 
                if is_best:
                    best_swa_rank5 = swa_test_result['rank5']
                    best_swa_rank5_epoch = epoch + 1

                is_best = swa_test_result['rank10'] > best_swa_rank10 
                if is_best:
                    best_swa_rank10 = swa_test_result['rank10']
                    best_swa_rank10_epoch = epoch + 1

                swa_mAP_list.append(swa_test_result['mAP'])
                swa_rank1_list.append(swa_test_result['rank1'])
                swa_rank5_list.append(swa_test_result['rank5'])
                swa_rank10_list.append(swa_test_result['rank10'])
                swa_epochs.append(epoch)

        if use_gpu:
            state_dict = model.module.state_dict()
            swa_state_dict = swa_model.module.state_dict() if args.swa else None
        else:
            state_dict = model.state_dict()
            swa_state_dict = swa_model.state_dict() if args.swa else None
        swa_n = swa_n if args.swa else None

        best_mAP = max(best_init_mAP, best_swa_mAP)
        best_mAP_epoch = max(best_init_mAP_epoch, best_swa_mAP_epoch)

        best_rank1 = max(best_init_rank1, best_swa_rank1)
        best_rank1_epoch = max(best_init_rank1_epoch, best_swa_rank1_epoch)

        best_rank5 = max(best_init_rank5, best_swa_rank5)
        best_rank5_epoch = max(best_init_rank5_epoch, best_swa_rank5_epoch)

        best_rank10 = max(best_init_rank10, best_swa_rank10)
        best_rank10_epoch = max(best_init_rank10_epoch, best_swa_rank10_epoch)

        best_mAP_list.append(best_mAP)
        best_rank1_list.append(best_rank1)
        best_rank5_list.append(best_rank5)
        best_rank10_list.append(best_rank10)

        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint({
                'state_dict': state_dict,
                'swa_state_dict': swa_state_dict,
                # 'mAP': test_result['mAP'],
                # 'rank1': test_result['rank1'],
                # 'swa_mAP': swa_test_result['mAP'],
                # 'swa_rank1': swa_test_result['rank1'],
                'epoch': epoch+1,
                # 'swa_n': swa_n,
                'optimizer': optimizer.state_dict(),
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

        x1 = np.linspace(1, len(Loss), len(Loss)) 
        ax1.set_title(label_1)
        # ax1.set_xticklabels(x1)
        # ax1.set_yticklabels(Loss)
        ax1.set_xlabel('iterations')
        ax1.set_ylabel('losses')
        ax1.grid(True, ls= '--')
        ax1.plot(x1, Loss, 'r-', mec='k', lw=1)    # x1  ->  epochs
        fig1.savefig('{}/train_loss_iteration.jpg'.format(args.save_dir))


        # x2 = np.linspace(1, len(init_mAP_list), len(init_mAP_list))
        ax2.set_title(label_2)
        # ax2.set_xticklabels(x2)
        # ax2.set_yticklabels(mAP_list)
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('mAP')
        ax2.grid(True, ls= '--')
        ax2.plot(init_epochs, init_mAP_list, 'b-', mec='k', lw=1)
        # ax2.plot(x2, init_mAP_list, 'b-', mec='k', lw=1)       
        fig2.savefig('{}/test_mAP(init)_epoch{}.jpg'.format(args.save_dir, args.max_epoch))


        # x3 = np.linspace(1, len(init_rank1_list), len(init_rank1_list))
        ax3.set_title(label_3)
        # ax3.set_xticklabels(x3)
        # ax3.set_yticklabels(rank1_list)
        ax3.set_xlabel('epochs')
        ax3.set_ylabel('rank')
        ax3.grid(True, ls= '--')
        ax3.plot(init_epochs, init_rank1_list, 'r-', mec='k', lw=1)   # x3  -> init_epochs
        ax3.plot(init_epochs, init_rank5_list, 'g-', mec='k', lw=1)
        ax3.plot(init_epochs, init_rank10_list, 'b-', mec='k', lw=1)
        fig3.savefig('{}/test_rank(init)_epoch{}.jpg'.format(args.save_dir, args.max_epoch))

        # x6 = np.linspace(1, len(best_mAP_list), len(best_mAP_list))
        ax6.set_title(label_6)
        # ax6.set_xticklabels(x6)
        # ax6.set_yticklabels(mAP_list)
        ax6.set_xlabel('epochs')
        ax6.set_ylabel('mAP')
        ax6.grid(True, ls= '--')
        ax6.plot(init_epochs, best_mAP_list, 'b-', mec='k', lw=1)
        fig6.savefig('{}/test_mAP(best)_epoch{}.jpg'.format(args.save_dir, args.max_epoch))


        # x7 = np.linspace(1, len(best_rank1_list), len(best_rank1_list))
        ax7.set_title(label_7)
        # ax7.set_xticklabels(x7)
        # ax7.set_yticklabels(rank1_list)
        ax7.set_xlabel('epochs')
        ax7.set_ylabel('rank')
        ax7.grid(True, ls= '--')
        ax7.plot(init_epochs, best_rank1_list, 'r-', mec='k', lw=1)
        ax7.plot(init_epochs, best_rank5_list, 'g-', mec='k', lw=1)
        ax7.plot(init_epochs, best_rank10_list, 'b-', mec='k', lw=1)
        fig7.savefig('{}/test_rank(best)_epoch{}.jpg'.format(args.save_dir, args.max_epoch))       

        if test_result['rank1'] != None:
            init_mAP_value = 'Epoch ' + str(epoch+1) + ' init_mAP ' + str(test_result['mAP']) + '\n'
            init_rank_value = 'Epoch ' + str(epoch+1) + ' init_rank1 ' + str(test_result['rank1']) + '\n' + \
                        'Epoch ' + str(epoch+1) + ' init_rank5 ' + str(test_result['rank5']) + '\n' + \
                        'Epoch ' + str(epoch+1) + ' init_rank10 ' + str(test_result['rank10']) + '\n' + \
                        '----------------' + '\n'
        
            init_mAP_txt.write(init_mAP_value)
            init_rank_txt.write(init_rank_value)

        if swa_test_result['rank1'] != None:
            swa_mAP_value = 'Epoch ' + str(epoch+1) + ' swa_mAP ' + str(swa_test_result['mAP']) + '\n'
            
            swa_rank_value = 'Epoch ' + str(epoch+1) + ' swa_rank1 ' + str(swa_test_result['rank1']) + '\n' + \
                        'Epoch ' + str(epoch+1) + ' swa_rank5 ' + str(swa_test_result['rank5']) + '\n' + \
                        'Epoch ' + str(epoch+1) + ' swa_rank10 ' + str(swa_test_result['rank10']) + '\n' + \
                        '----------------' + '\n'   
            
            swa_mAP_txt.write(swa_mAP_value)
            swa_rank_txt.write(swa_rank_value)


        best_mAP_value = 'Epoch ' + str(epoch+1) + ' best_mAP ' + str(best_mAP) + '\n'
        
        best_rank_value = 'Epoch ' + str(epoch+1) + ' best_rank1 ' + str(best_rank1) + '\n' + \
                    'Epoch ' + str(epoch+1) + ' best_rank5 ' + str(best_rank5) + '\n' + \
                    'Epoch ' + str(epoch+1) + ' best_rank10 ' + str(best_rank10) + '\n' + \
                    '----------------' + '\n'   
        
        best_mAP_txt.write(best_mAP_value)
        best_rank_txt.write(best_rank_value)
        
 
    init_mAP_txt.close()
    init_rank_txt.close()
    swa_mAP_txt.close()
    swa_rank_txt.close() 
    best_mAP_txt.close() 
    best_rank_txt.close()

   
    if args.max_epoch % args.save_freq != 0:
        save_checkpoint({
                'state_dict': state_dict,
                'swa_state_dict': swa_state_dict,
                # 'mAP': test_result['mAP'],
                # 'rank1': test_result['rank1'],
                # 'swa_mAP': swa_test_result['mAP'],
                # 'swa_rank1': swa_test_result['rank1'],
                'epoch': args.max_epoch,
                # 'swa_n': swa_n,
                'optimizer': optimizer.state_dict()
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))
 
    # x0 = np.linspace(1, args.max_epoch, args.max_epoch)
    ax0.set_title(label_0)
    # ax0.set_xticklabels(x0)
    # ax0.set_yticklabels(Loss_every_epoch)
    ax0.set_xlabel('epochs')
    ax0.set_ylabel('losses')
    ax0.grid(True, ls= '--')
    ax0.plot(epochs, Loss_every_epoch, 'r-', mec='k', label=labels_1[0], lw=1)
    ax0.legend(loc = 'best', fontsize=8)
    fig0.savefig('{}/train_loss_epoch{}.jpg'.format(args.save_dir, args.max_epoch))

    ax1.plot(x1, Loss, 'r-', mec='k', label=labels_1[0], lw=1)
    ax1.legend(loc = 'best', fontsize=8)
    fig1.savefig('{}/train_loss_iteration.jpg'.format(args.save_dir))

    ax2.plot(init_epochs, init_mAP_list, 'b-', mec='k', label=labels_2[0], lw=1)
    ax2.legend(loc = 'best', fontsize=8)
    fig2.savefig('{}/test_mAP(init)_epoch{}.jpg'.format(args.save_dir, args.max_epoch))

    # ax3.plot(x3, init_rank1_list, 'b-', mec='k', label=labels_3[0], lw=1)
    ax3.legend((labels_3[0], labels_3[1], labels_3[2]), loc = 'best', fontsize=8)
    fig3.savefig('{}/test_rank(init)_epoch{}.jpg'.format(args.save_dir, args.max_epoch))

    # x4 = np.linspace(args.swa_start, args.max_epoch, (args.max_epoch - args.swa_start + 1))
    ax4.set_title(label_4)
    # ax4.set_xticklabels(x4)
    # ax4.set_yticklabels(swa_mAP_list)
    ax4.set_xlabel('epochs')
    ax4.set_ylabel('mAP')
    ax4.grid(True, ls= '--')
    ax4.plot(swa_epochs, swa_mAP_list, 'b-', mec='k', label=labels_4[0], lw=1)
    ax4.legend(loc = 'best', fontsize=8)
    fig4.savefig('{}/test_mAP(swa)_epoch{}.jpg'.format(args.save_dir, args.max_epoch))

    # x5 = np.linspace(args.swa_start, args.max_epoch, (args.max_epoch - args.swa_start + 1))
    ax5.set_title(label_5)
    # ax5.set_xticklabels(x5)
    # ax5.set_yticklabels(swa_rank1_list)
    ax5.set_xlabel('epochs')
    ax5.set_ylabel('rank')
    ax5.grid(True, ls= '--')
    ax5.plot(swa_epochs, swa_rank1_list, 'r-', mec='k', label=labels_5[0], lw=1)
    ax5.plot(swa_epochs, swa_rank5_list, 'g-', mec='k', label=labels_5[1], lw=1)
    ax5.plot(swa_epochs, swa_rank10_list, 'b-', mec='k', label=labels_5[2], lw=1)
    ax5.legend(loc = 'best', fontsize=8)
    fig5.savefig('{}/test_rank(swa)_epoch{}.jpg'.format(args.save_dir, args.max_epoch))

    ax6.plot(init_epochs, best_mAP_list, 'b-', mec='k', label=labels_6[0], lw=1)
    ax6.legend(loc = 'best', fontsize=8)
    fig6.savefig('{}/test_mAP(best)_epoch{}.jpg'.format(args.save_dir, args.max_epoch))

    # ax7.plot(x7, best_rank1_list, 'b-', mec='k', label=labels_7[0], lw=1)
    ax7.legend((labels_7[0], labels_7[1], labels_7[2]),loc = 'best', fontsize=8)
    fig7.savefig('{}/test_rank(best)_epoch{}.jpg'.format(args.save_dir, args.max_epoch))

    print("==> Best Init mAP {:.1%}, achieved at epoch {}".format(best_init_mAP, best_init_mAP_epoch))
    print("==> Best Swa mAP {:.1%}, achieved at epoch {}".format(best_swa_mAP, best_swa_mAP_epoch))
    print("==> Best mAP {:.1%}, achieved at epoch {}".format(best_mAP, best_mAP_epoch)) 
    print('------------------------------------------------------------------------------')

    print("==> Best Init Rank-1 {:.1%}, achieved at epoch {}".format(best_init_rank1, best_init_rank1_epoch))
    print("==> Best Swa Rank-1 {:.1%}, achieved at epoch {}".format(best_swa_rank1, best_swa_rank1_epoch))
    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_rank1_epoch))
    print('------------------------------------------------------------------------------')

    print("==> Best Init Rank-5 {:.1%}, achieved at epoch {}".format(best_init_rank5, best_init_rank5_epoch))
    print("==> Best Swa Rank-5 {:.1%}, achieved at epoch {}".format(best_swa_rank5, best_swa_rank5_epoch))
    print("==> Best Rank-5 {:.1%}, achieved at epoch {}".format(best_rank5, best_rank5_epoch))
    print('------------------------------------------------------------------------------')

    print("==> Best Init Rank-10 {:.1%}, achieved at epoch {}".format(best_init_rank10, best_init_rank10_epoch))
    print("==> Best Swa Rank-10 {:.1%}, achieved at epoch {}".format(best_swa_rank10, best_swa_rank10_epoch))
    print("==> Best Rank-10 {:.1%}, achieved at epoch {}".format(best_rank10, best_rank10_epoch))
    print('------------------------------------------------------------------------------')

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

# def accuracy(output, target, topk=(1,)):
#     """Computes the prec@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res

def train(epoch, model, criterion_class, criterion_metric, optimizer, trainloader, use_gpu):
    model.train()
    # 初始化损失
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    xent_losses = AverageMeter()
    global_losses = AverageMeter()
    local_losses = AverageMeter()
    # top1 = AverageMeter()
    # correct = 0

    loss_txt = open(osp.join(args.save_dir, 'loss({}).txt'.format(args.dataset)), 'a+')

    end = time.time()
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for循环当中。
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):     # 每一步，trainloader都释放一小批数据用来学习
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        # imgs, pids = torch.autograd.Variable(imgs), torch.autograd.Variable(pids)
        # measure data loading time
        data_time.update(time.time() - end)
        # 输入 imgs，经过模型处理，得到输出。outputs: tensor
        outputs, features, local_features = model(imgs)        
        #prec1, prec5 = accuracy(outputs.data, pids, topk=(1,5))     
        # pred = outputs.data.max(1, keepdim=True)[1]

        if args.htri_only:
            if isinstance(features, tuple):
                global_loss, local_loss = DeepSupervision(criterion_metric, features, pids, local_features)
            else:
                global_loss, local_loss = criterion_metric(features, pids, local_features)
        else:
            if isinstance(outputs, tuple):
                # xent_loss：交叉熵损失
                xent_loss = DeepSupervision(criterion_class, outputs, pids)
            else:
                # 计算分类损失，得到交叉熵损失，仅计算全局的损失
                # Cross entropy loss means cross entropy global loss.
                xent_loss = criterion_class(outputs, pids)

            # For the triplet, the loss is computed based on both the global distance and the local distance margins (p.3).
            # The metric loss is decided by both the global distances and the local distances (p.4).
            # Global loss means triplet global loss, and local loss means triplet local loss.
            if isinstance(features, tuple):
                global_loss, local_loss = DeepSupervision(criterion_metric, features, pids, local_features)
            else:
                # 计算度量损失，得到三元组损失。计算了全局和局部的损失
                # global_loss, local_loss = criterion_metric(features, pids, local_features)
                global_loss, local_loss = criterion_metric(features, pids, local_features)
        loss = xent_loss + global_loss + local_loss    # 计算总的损失
        optimizer.zero_grad()        # 清空上一步的残余更新参数值
        loss.backward()              # 误差反向传播, 计算参数更新值
        optimizer.step()             # 将参数更新值施加到 net 的 parameters 上

        batch_time.update(time.time() - end)
        end = time.time()
        # x.size(0)指batchsize的值
        losses.update(loss.item(), pids.size(0))
        xent_losses.update(xent_loss.item(), pids.size(0))
        global_losses.update(global_loss.item(), pids.size(0))
        local_losses.update(local_loss.item(), pids.size(0))

        #top1.update(prec1[0], pids.size(0))
        # correct += (pred == pids).sum().item()
        # correct  += torch.sum(pred  == pids.data)
        # pred = torch.max(outputs.data, 1)[1]
        # correct += pred.eq(pids.data.view_as(pred)).sum().item()

        Loss.append(losses.val)

	    #'Top1 {top1.val:.3f} ({top1.avg:.3f})\t','Top5 {top5.val:.3f} ({top5.avg:.3f})', top1=top1, top5=top5
	    #100.0*(batch_idx+1)/len(trainloader) 
        if (batch_idx+1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'CLoss {xent_loss.val:.3f} ({xent_loss.avg:.3f})\t'
                  'GLoss {global_loss.val:.3f} ({global_loss.avg:.3f})\t'
                  'LLoss {local_loss.val:.3f} ({local_loss.avg:.3f})\t'.format(
                   epoch+1, batch_idx+1, len(trainloader), batch_time=batch_time, data_time=data_time,      
                   loss=losses,xent_loss=xent_losses, global_loss=global_losses, local_loss=local_losses)
                   )  

    Loss_every_epoch.append(losses.avg)

    loss_value = 'Epoch ' + str(epoch+1) + ' loss ' + str(losses.avg) + '\n'
    loss_txt.write(loss_value)
    loss_txt.close()

    return {'loss':losses.val}
    # return{'loss_val': '{loss.val:.4f}'.format(loss=losses), 'loss_avg': '{loss.avg:.4f}'.format(loss=losses)}

def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], Swa=True):
    batch_time = AverageMeter()

    model.eval()
    # 提取特征
    # qf: query features
    # lqf: local query features
    with torch.no_grad():
        qf, q_pids, q_camids, lqf = [], [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features, local_features = model(imgs)
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
    qf = 1. * qf / (torch.norm(qf, 2, dim = -1, keepdim=True).expand_as(qf) + 1e-12)
    gf = 1. * gf / (torch.norm(gf, 2, dim = -1, keepdim=True).expand_as(gf) + 1e-12)
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    if args.dataset != 'veri':
        q_camids = np.zeros(m).astype(np.int32)
        g_camids = np.ones(n).astype(np.int32)

    # print('g_camids', g_camids)
    
    if not args.test_distance== 'global':
        print("-------------------")
        print("Only using global branch")
        from util.distance import low_memory_local_dist
        lqf = lqf.permute(0,2,1)
        lgf = lgf.permute(0,2,1)
        local_distmat = low_memory_local_dist(lqf.numpy(),lgf.numpy(),aligned= not args.unaligned)
        if args.test_distance== 'local':
            print("-------------------")
            print("Only using local branch")
            distmat = local_distmat
        if args.test_distance == 'global_local':
            print("-------------------")
            print("Using global and local branches")
            distmat = local_distmat+distmat

    # print("-------------------")
    # print("Computing CMC and mAP")
    # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)
    # print("Results ----------")
    # print("mAP: {:.1%}".format(mAP))
    # print("CMC curve")
    # for r in ranks:
    #     print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    # print("------------------")

    if args.reranking:
        from util.re_ranking import re_ranking
        if args.test_distance == 'global':
            print("-------------------")
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
                print("-------------------")
                print("Only using local branch for reranking")
                distmat = re_ranking(qf,gf,k1=20,k2=6,lambda_value=0.3,local_distmat=local_dist,only_local=True)
            elif args.test_distance == 'global_local':
                print("-------------------")
                print("Using global and local branches for reranking")
                distmat = re_ranking(qf,gf,k1=20,k2=6,lambda_value=0.3,local_distmat=local_dist,only_local=False)
        print("Computing CMC and mAP for re_ranking")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)


        print("Results ----------")
        if not Swa:
            print("mAP(RK): {:.1%}".format(mAP))
            print("CMC curve(RK)")
            for r in ranks:
                print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))

        else:
            print("Swa mAP(RK): {:.1%}".format(mAP))
            print("Swa CMC curve(RK)")
            for r in ranks:
                print("Swa Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))


        print("------------------")
    return {'rank1': cmc[0], 'rank5': cmc[4], 'rank10': cmc[9], 'mAP': mAP}

if __name__ == '__main__':
    main()
