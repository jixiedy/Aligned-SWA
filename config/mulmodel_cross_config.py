#coding=utf-8
from __future__ import absolute_import

import argparse
import util.data_manager as data_manager
import models

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

parser.add_argument('--use_metric_noncamid', action='store_true',
                    help="whether to use noncamid_metric (default: False)")

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
# parser.add_argument('--optim2', type=str, default='swa', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--swa_epoch', default=161, type=int, help="optimization algorithm (see optimizers.py)")
# 迭代终止时的那个 epoch，不管从第几个 epoch 开始，都计算到这个 epoch 就停止
parser.add_argument('--max_epoch', default=300, type=int, help="maximum epochs to run")     
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
# parser.add_argument('--eval_step', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
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

# Architecture 使用哪个神经网络模型
# parser.add_argument('-a1', '--arch1', type=str, default='densenet121', choices=models.get_names())     
parser.add_argument('-a1', '--arch1', type=str, default='condensenet', choices=models.get_names())     
parser.add_argument('-a2', '--arch2', type=str, default='resnet50', choices=models.get_names())    
parser.add_argument('-a3', '--arch3', type=str, default='densenet161', choices=models.get_names())    
parser.add_argument('-a4', '--arch4', type=str, default='inceptionv4', choices=models.get_names())    
parser.add_argument('-a5', '--arch5', type=str, default='densenet169', choices=models.get_names())    
parser.add_argument('-a6', '--arch6', type=str, default='resnet152', choices=models.get_names())    
parser.add_argument('-a7', '--arch7', type=str, default='densenet201', choices=models.get_names())    
parser.add_argument('-a8', '--arch8', type=str, default='resnet101', choices=models.get_names())    
parser.add_argument('-a9', '--arch9', type=str, default='densenet121', choices=models.get_names())
parser.add_argument('-a10', '--arch10', type=str, default='shufflenetv2', choices=models.get_names())     
parser.add_argument('-a11', '--arch11', type=str, default='shufflenet', choices=models.get_names())    

# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")

# resume:训练好的模型所在的路径，测试时需要从该路径下导入模型。如果想接着上次的模型继续训练，可以从这里导入上次的模型
parser.add_argument('--resume', type=str, default='', metavar='PATH')
# parser.add_argument('--resume', type=str, default=None, metavar='checkpoint', 
                    # help='checkpoint to resume training from (default: None)')
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

parser = argparse.ArgumentParser(description='condensenet option')

parser.add_argument('--stages', type=str, metavar='STAGE DEPTH',help='per layer depth')
parser.add_argument('--bottleneck', default=4, type=int, metavar='B',help='bottleneck (default: 4)')
parser.add_argument('--group-1x1', type=int, metavar='G', default=4,help='1x1 group convolution (default: 4)')
parser.add_argument('--group-3x3', type=int, metavar='G', default=4,help='3x3 group convolution (default: 4)')
parser.add_argument('--condense-factor', type=int, metavar='C', default=4, help='condense factor (default: 4)')
parser.add_argument('--growth', type=str, metavar='GROWTH RATE',help='per layer growth')
# parser.add_argument('--reduction', default=0.5, type=float, metavar='R',help='transition reduction (default: 0.5)')
parser.add_argument('--dropout-rate', default=0, type=float,help='drop out (default: 0)')

args = parser.parse_args()

args.stages = list(map(int, args.stages.split('-')))
args.growth = list(map(int, args.growth.split('-')))
if args.condense_factor is None:
    args.condense_factor = args.group_1x1

# 返回对象object的属性和属性值的字典对象，如果没有参数，就打印当前调用位置的属性和属性值, 类似 locals()。
for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False