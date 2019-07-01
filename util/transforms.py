#coding=utf-8
"""
关于这句from __future__ import absolute_import的作用:
直观地看就是说”加入绝对引入这个新特性”。说到绝对引入，当然就会想到相对引入。那么什么是相对引入呢?比如说，你的包结构是这样的:
pkg/
pkg/init.py
pkg/main.py
pkg/string.py

如果在main.py中写import string,Python会先查找当前目录下有没有string.py,若找到了，则引入该模块，
然后在main.py中可以直接用string了。如果你是真的想用同目录下的string.py那就好，但是如果你是想用系统自带的标准string.py呢？
那其实没有什么好的简洁的方式可以忽略掉同目录的string.py而引入系统自带的标准string.py。这时候你就需要from __future__ import absolute_import了。
这样，你就可以用import string来引入系统的标准string.py, 而用from pkg import string来引入当前目录下的string.py了.
"""
from __future__ import absolute_import
from torchvision.transforms import *
from PIL import Image
import random
import numpy as np

class Random2DTranslation(object):

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
            Args:
                img (PIL Image): Image to be cropped.

            Returns:
                PIL Image: Cropped image.
        """
        if random.random() < self.p:
            return img.resize((self.width, self.height), self.interpolation)
        # round()方法返回浮点数x的四舍五入值。
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        # uniform()方法将随机生成下一个实数，它在 [x, y) 范围内。
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img

if __name__ == '__main__':
    pass