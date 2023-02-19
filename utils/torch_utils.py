# -------------------------------
# -*- coding = utf-8 -*-
# @Time : 2023/2/11 20:59
# @Author : chc_stars
# @File : torch_utils.py
# @Software : PyCharm
# -------------------------------
"""
Pytorch utils
"""

import datetime
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from utils.general import LOGGER

try:
    import thop  # 用于FLOPS计算
except ImportError:
    thop = None


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator使分布式培训中的所有进程等待每个local_master执行某些操作。
    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])

def date_modified(path=__file__):
    # 返回可读文件修改日期，即“2021 - 3 - 26”
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'

def git_describe(path=Path(__file__).parent):  # path must be a directory
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''  # not a git repository


def select_device(device='', batch_size=None):
    # device = ' cpu'  or '0' or '0, 1, 2, 3'
    s = f'YOLOv5 {git_describe() or date_modified()} torch {torch.__version__}'        # string
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0'  to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # force torch.cuda.is_available() = False
    elif device:  # 请求的非cpu设备
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device}  requested'   # 检查可用性

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'   # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # GPU数
        if n > 1 and batch_size:  # 检查batch_size 是否能被 GPU数整除
            assert batch_size % n == 0, f'batch_size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space } CUDA:{d} ({p.name},  {p.total_memory / 1024 ** 2:.0f}MiB)\n"    # bytes to MB
    else:
        s += 'CPU\n'

    LOGGER.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)   # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(input, ops, n=10, device=None):
    # YOLOv5速度 / 内存 / FLOP 分析器

    # Usage:
    #  input = torch.randn(16, 3, 640, 640)
    #  m1 = lamba x: x* torch.sigmoid(x)
    #  m2 = nn.SiLU()
    #  profile(input, [m1, m2], n = 100)   # profile over 100 iterarions

    results = []
    device = device or select_device()

    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>24s}")

    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.required_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, 'to') else m # device
            m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward

            try:
                flops = thop.profile(m, inputs=(x, ), verbose=False)[0] / 1E9 * 2  # GFLOps

            except:
                flops = 0

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception as e:              # no backwaed method
                        # print(e)  # for debug
                        t[2] = float('nan')
                    tf += (t[1] - t[0]) * 1000 / n   # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n   # ms per op backward

                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available()  else 0  # (GB)
                s_in = tuple(x.shape) if isinstance(x, torch.Tensor)  else 'list'
                s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else 'list'
                p = sum(list(x.numel() for x in m.parameters())) if isinstance(m, nn.Module) else 0 # parameters
                print(f'{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}')
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results


def is_parallel(model):
    # 返回真 如果模型类型是DP或者DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
   #  去并行化模型：如果模型是DP或DDP类型，则返回单个GPU模型
    return model.module if is_parallel(model) else model


def initialize_weight(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass      #  nn.init.kaiming_normal_(m.weight, mode='fan_out, nonlinearity='rule)
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SELU]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # Finds layers indices matching module class 'mclass'  查找与模块类“mclass”匹配的层索引
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    #  返回全局模型稀疏性
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    # 修剪模型以请求全局稀疏性
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent

    print(' % .3g global spasity' % sparsity(model))


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # 准备空间偏差
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, img_size=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)   # number gradient
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters': >12} {'shape':>20} {'mu': >10} {'sigma': >10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:   # FLOPs
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.ymal.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        img_size = img_size if  isinstance(img_size, list) else [img_size, img_size]  # expand if int / float
        fs = ', %  GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640*640 GFLOPs
    except (ImportError, Exception):
        fs =''

    LOGGER.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def load_classifier(name='resnet101', n=2):
    # Load a pretrained model reshape to n-class output
    model = torchvision.models.__dict__[name](pretrained=True)

    # ResNet model properties
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Reshape output to n classes
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model


def scale_img(img, ratio=1.0, same_shape=False, gs=32):   # img(16, 3, 256, 416)
    # scales img (bs, 2, y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w, = img.shape[2:]
        s  = (int(h * ratio), int(w * ratio)) # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad / crop img
            h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
        return F.pad(img, [0, w-s[1], 0, h - s[0]], value=0.447)   # value = imagemet mean


def copy_attr(a, b, include=(), exclude=()):
    # # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class EarlyStopping:
    #
    def __init__(self, patience = 30):
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float('inf')   # epoch to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur epoch

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness: # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch- self.best_epoch # epochs without imporvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded

        if stop:
            LOGGER.info(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                        f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n'
                        f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                        f'i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.')

        return stop

class ModelEMA:
    """模型指数移动平均值https: // github.com / rwightman / pytorch - image - models
    保持模型state_dict（参数和缓冲区）中所有内容的移动平均值。
    这是为了允许以下功能
    https: // www.tensorflow.org / api_docs / python / tf / train / ExponentialMovingAverage
    为了使某些训练方案表现良好，权重的平滑版本是必要的。
    这个类是敏感的，它是在model
    init的序列中初始化的，
    GPU分配和分布式训练包装。
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Crete EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates +=1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


if __name__ == '__main__':
    pass