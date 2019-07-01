from __future__ import absolute_import

from .ResNet import *
from .DenseNet import *
from .ShuffleNet import *
from .ShuffleNetV2 import *
from .InceptionV4 import *
# from .CondenseNet_Other import *
from .CondenseNet import *
from .resnet_cbam import *
from .resnet_cbam_2 import *
from .densenet_cbam import *

__factory = {
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
    'densenet121': DenseNet121,
    'densenet161': DenseNet161,
    'densenet169': DenseNet169,
    'densenet201': DenseNet201,
    'condensenet': CondenseNet,
    # 'condensenet_oth': CondenseNet_Other,
    'shufflenet': ShuffleNet,
    'shufflenetv2': ShuffleNetV2,
    'inceptionv4': InceptionV4ReID,
    'resnet50_cbam': resnet50_cbam,
    'resnet50_cbam_2': resnet50_cbam_2,
    'densenet121_cbam': densenet121_cbam,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)