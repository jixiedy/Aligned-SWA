# -*- coding: utf-8 -*-
import torch
from torch.optim import Optimizer
import math
# from torchsample.callbacks import Callback

# __all__：暴露接口，是对于模块公开接口的一种约定
# 提供了暴露接口用的”白名单“。一些不以下划线开头的变量（比如从其他地方 import 到当前模块的成员）可以同样被排除出去
__all__ = ['init_optim']

"""
weight_decay（权值衰减）使用的目的是防止过拟合。在损失函数中，weight_decay是放在正则项（regularization）前面的一个系数，
正则项一般指示模型的复杂度，所以weight_decay的作用是调节模型复杂度对损失函数的影响，若weight_decay很大，则复杂的模型损失函数的值也就大。

momentum是梯度下降法中一种常用的加速技术.对于一般的SGD，其表达式为: x←x − α∗dx.
x沿负梯度方向下降.
而带momentum项的SGD则写生如下形式：
v=β∗v−a∗dx
x←x+v
其中 β 即momentum系数，通俗的理解上面式子就是，如果上一次的momentum（即v）与这一次的负梯度方向是相同的，那这次下降的幅度就会加大，
所以这样做能够达到加速收敛的过程。

"""
def init_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'adamax':
        return torch.optim.Adamax(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'asgd':
        return torch.optim.ASGD(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    elif optim == 'adadelta':
        return torch.optim.Adadelta(params, lr =lr, weight_decay=weight_decay)
    elif optim == 'adagrad':
        return torch.optim.Adagrad(params, lr =lr, weight_decay=weight_decay)

    elif optim == 'nadam':
        # return torch.optim.Nadam(params, lr=lr, weight_decay=weight_decay)
        return Nadam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported optim: {}".format(optim))

"""Implements Nadam algorithm (a variant of Adam based on Nesterov momentum).
    It has been proposed in `Incorporating Nesterov Momentum into Adam`__.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        schedule_decay (float, optional): momentum schedule decay (default: 4e-3)
    __ http://cs229.stanford.edu/proj2015/054_report.pdf
    __ http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
"""

class Nadam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.004,amsgrad=False):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        amsgrad=amsgrad,weight_decay=weight_decay)
        super(Nadam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Nadam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    state['m_weight'] = 1
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']


                state['step'] += 1
                momentum_cache_t = beta1 * (
                        1. - 0.5 * math.pow(0.96, state['step'] * group['weight_decay'] ))
                momentum_cache_t_1 = beta1 * (
                        1. - 0.5 * math.pow(0.96, (state['step']+1) * group['weight_decay'] ))
                state['m_weight'] = state['m_weight'] * momentum_cache_t

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                m_t_prime = exp_avg/(1 - state['m_weight'] * momentum_cache_t_1)

                g_prime = grad.div(1 - state['m_weight'])
                m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq , out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    v_t_prime = max_exp_avg_sq/(1 - beta2 ** state['step'])
                else:
                    v_t_prime = exp_avg_sq / (1 - beta2 ** state['step'])

                denom = v_t_prime.sqrt().add_(group['eps'])
                p.data.addcdiv_(-group['lr'], m_t_bar , denom)

        return loss

"""
Stochastic Weight Averaging: https://arxiv.org/abs/1803.05407

Implementaton in Keras from user defined epochs assuming constant 
learning rate

Cyclic learning rate implementation in https://arxiv.org/abs/1803.05407 
not implemented

Created on July 4, 2018

@author: Krist Papadopoulos
"""

# class SWA(keras.callbacks.Callback):
# class SWA(Callback):   
#     def __init__(self, filepath, swa_epoch):
#         super(SWA, self).__init__()
#         self.filepath = filepath
#         self.swa_epoch = swa_epoch 
    
#     def on_train_begin(self, logs=None):
#         self.nb_epoch = self.params['epochs']
#         print('Stochastic weight averaging selected for last {} epochs.'
#               .format(self.nb_epoch - self.swa_epoch))
        
#     def on_epoch_end(self, epoch, logs=None):     
#         if epoch == self.swa_epoch:
#             self.swa_weights = self.trainer.get_weights()
            
#         elif epoch > self.swa_epoch:    
#             for i, layer in enumerate(self.trainer.layers):
#                 self.swa_weights[i] = (self.swa_weights[i] * 
#                     (epoch - self.swa_epoch) + self.trainer.get_weights()[i])/((epoch - self.swa_epoch)  + 1)  

#         else:
#             pass
        
#     def on_train_end(self, logs=None):
#         self.trainer.set_weights(self.swa_weights)
#         print('Final trainer parameters set to stochastic weight average.')
#         self.trainer.save_weights(self.filepath)
#         print('Final stochastic averaged weights saved to file.')

# class Swa(Callback):
#     def __init__(self, model, swa_model, swa_start):
#         super().__init__()
#         self.model,self.swa_model,self.swa_start=model,swa_model,swa_start
        
#     def on_train_begin(self):
#         self.epoch = 0
#         self.swa_n = 0

#     def on_epoch_end(self, metrics):
#         if (self.epoch + 1) >= self.swa_start:
#             self.update_average_model()
#             self.swa_n += 1
            
#         self.epoch += 1
            
#     def update_average_model(self):
#         # update running average of parameters
#         model_params = self.model.parameters()
#         swa_params = self.swa_model.parameters()
#         for model_param, swa_param in zip(model_params, swa_params):
#             swa_param.data *= self.swa_n
#             swa_param.data += model_param.data
#             swa_param.data /= (self.swa_n + 1)  