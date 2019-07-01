#coding=utf-8
from __future__ import absolute_import

import argparse

parser = argparse.ArgumentParser(description='condensenet option')

parser.add_argument('--stages', type=str, metavar='STAGE DEPTH',
                    help='per layer depth')
parser.add_argument('--bottleneck', default=4, type=int, metavar='B',
                    help='bottleneck (default: 4)')
parser.add_argument('--group-1x1', type=int, metavar='G', default=4,
                    help='1x1 group convolution (default: 4)')
parser.add_argument('--group-3x3', type=int, metavar='G', default=4,
                    help='3x3 group convolution (default: 4)')
parser.add_argument('--condense-factor', type=int, metavar='C', default=4,
                    help='condense factor (default: 4)')
parser.add_argument('--growth', type=str, metavar='GROWTH RATE',
                    help='per layer growth')
# parser.add_argument('--reduction', default=0.5, type=float, metavar='R',
#                     help='transition reduction (default: 0.5)')
parser.add_argument('--dropout-rate', default=0, type=float,
                    help='drop out (default: 0)')

args = parser.parse_args()

args.stages = list(map(int, args.stages.split('-')))
args.growth = list(map(int, args.growth.split('-')))
if args.condense_factor is None:
    args.condense_factor = args.group_1x1

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False