#coding:utf-8
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm 
from matplotlib import axes
import logging
import sys

logger = logging.getLogger('Aligned-heatmaps')


# def draw_heatmap(imgs=None, outputs=None, features=None, local_features=None):
    # if imgs is None:
    #     logger.error('Image can not be read, path=%s' % imgs)
    #     sys.exit(-1)

    # fig = plt.figure()

    # ax = fig.add_subplot(2, 2, 1)
    # ax.set_title('img')
    # plt.imshow(imgs, alpha=0.5)


    # ax = fig.add_subplot(2, 2, 2)
    # ax.set_title('output')
    # plt.imshow(outputs, cmap=cm.jet, alpha=0.5)
    # plt.colorbar()

    # ax = fig.add_subplot(2, 2, 3)
    # ax.set_title('features')
    # plt.imshow(features, cmap=cm.jet, alpha=0.5)
    # plt.colorbar()


    # ax = fig.add_subplot(2, 2, 4) 
    # ax.set_title('local_features')
    # plt.imshow(local_features, cmap=cm.jet, alpha=0.5)
    # plt.colorbar()

def draw_heatmap(inputs):

    fig = plt.figure()
    plt.imshow(inputs, cmap=cm.jet, alpha=0.5)
    plt.colorbar()
