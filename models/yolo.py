# -------------------------------
# -*- coding = utf-8 -*-
# @Time : 2023/2/13 10:56
# @Author : chc_stars
# @File : yolo.py
# @Software : PyCharm
# -------------------------------
import argparse
import sys
from copy import deepcopy
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  #增加路径到的PATH


from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (copy_attr, fuse_conv_and_bn, initialize_weight, model_info, scale_img, select_device, time_sync)

try:
    import thop   # for FLOPs computation
except ImportError:

    thop = None


class Detect(nn.Module):
    stride = None   # strides computed during building
    onnx_dynamic = False   # onnx export parameter

    def __init__(self, nc, anchors=(), ch=(), inplace=True):  # 检测层
        super().__init__()
        self.nc = nc   # 类别数
        self.no = nc + 5  # 每个锚点的输出数
        self.nl = len(anchors)  # 检测层数
        self.na = len(anchors[0]) // 2  # 锚框数
        self.grid = [torch.zeros(1)] * self.nl # 初始化网格
        self.anchor_grid = [torch.zeros(1)] * self.nl   # 初始化锚框网格
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl, na, 2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)   # 输出卷积
        self.inplace = inplace  # #使用就地操作（例如切片分配）

    def forward(self, x):
        z = []  # 推断输出
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  #  x(bs, 255, 20, 20 to x(bs, 3, 20, 20, 85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training: # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2  - 0.5 + self.grid[i]) * self.stride[i]   #xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:    # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]   # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):  # 生成网格
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')

        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):   # 模型， 输入通道， 类别数
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg            # 模型字典
        else:  # is *.yaml
            import yaml   # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)    # 模型字典

        #  定义模型
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)    # 输入通道
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # 覆盖 yaml值
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])        # model,  savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]    # 默认名字
        self.inplace = self.yaml.get('inplace', True)

        #  建立步长， 锚框
        m = self.model[-1]   # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])   # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()   # only run once

        # Init weight, biases
        initialize_weight(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)   # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]   # h, w
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)

        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # output
        for m in self.model:
            if m.f != -1:   # 如果不是来自上一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)  # 保存输出
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # 增强推理后的去尺度预测（逆运算）, 将推理结果恢复到原图图像尺寸（逆操作）
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale # de-scale
            if flips == 2:
                y = img_size[0] - y # de-flip  ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr

            p = torch.cat((x, y, wh, p[..., 4:]), -1)

        return p

    def _clip_augmented(self, y):  # 这个是TTA的时候对原图像进行裁剪， 也是一种数据增强方式，用在TTA测试的时候
        nl = self.model[-1].nl  #  number of detection layers (p3-p5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1 # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))   # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl -1 -x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:] # small
        return y

    def _profile_one_layer(self, m, x, dt):  # 打印日志，前向推理时间
        c = isinstance(m, Detect)   #
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs': >10s}{'params':>10s}   {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f}  {o:10.2f}  {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):   # initialize biases into Detect(), cf ius class frequency
        ## https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3, 85)
            b.data[:, 4] += math.log(8 / (640 / s) **2)  # obj(8 objects per image)
            b.data[:, 5] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):  # 打印模型中最后Detect模块里面的卷积层的偏置信息（也可以任选那些层偏置信息）
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T          #  conv.bias(255)  to (3, 85)
            LOGGER.info(
                ("%6g Conv2d.bias:" + '%10.3g' * 6) % ( mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean())
            )
    #
    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('% 10.3g'  %(m.w.detach().sigmoid() * 2))      # shorcut weight

    # fuse() 是用来进行conv和bn层合并，为了提速模型推理速度
    def fuse(self):   # fuse model Conv2d() + BN layer
        """
        用在detect.py和val.py
        fuse model Conv2d（） + BatchNorm2d() layers
        调用 oneflow_utils.py中的fuse_conv_and_bn函数和common.py 中Conv 模块的fuseforward函数
        """
        LOGGER.info('Fusing layers...')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):   # 若当前层有conv和bn 就调用fuse_conv_and_bn函数
                m.conv = fuse_conv_and_bn(m.conv, m.bn)   # update conv
                delattr(m, 'bn')   # 移除bn
                m.forward = m.forward_fuse  # 更新前向传播
        self.info()
        return self

    def autoshape(self):  # add AutoShape module
        LOGGER.info('Adding AutoShape...')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attribures
        return m

    def info(self, verbose=False, img_size=640): # 打印模型信息
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # 将to（）、cpu（）、cuda（）、half（）应用于非参数或注册缓冲区的模型张量
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


#  解析网络模型配置并构建模型
def parse_model(d, ch):        # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # 锚框数
    no = na * (nc + 5)   # 输出数  = 锚框数 * （类别数 + 5）

    # 开始搭建网络
    # layers: 保存每一层的层结构
    # save: 记录下所有层结构中from中不是-1的层结构序号
    # c2: 保存当前层的输出channel
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # enumerate() 用于将一个可遍历的数据对象（如列表，元组）组合为一个索引序列，同时列出数据和数据下标
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from , number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain  # 将深度和深度因子相乘， 计算层深度，深度最小为1
        if m in [Conv, GhostConv, Bottleneck, GhostBottlenck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            c1, c2 = ch[f], args[0]      # c1： 输入通道， c2 : 输出通道
            if c2 != no: # if not output 如果不是最后一层，则将通道数乘以宽度因子，即宽度因子作用与最后一层之外的所有层
                c2 = make_divisible(c2 * gw, 8)   # make-divisible使得原始通道乘以宽度因子之后取整到8的倍数，让模型更好的并行

            args = [c1, c2, *args[1:]]    # 将前面的结果保存在args中， 也就是这个模块最终的输出参数
            # 根据每层网络参数的不同，分别处理参数，具体各个类的参数是什么请参考他们的——init——方法
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # 锚框数
                args[1] = [list(range(args[1] * 2))] * len(f)

        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        # 构建整个网络模块，这里就是根据模块的重复次数n以及模块本身和他的参数来构建这个模块和参数对应的Module
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)   # module
        # 获取模块(module type)具体名例如 models.common.Conv , models.common.C3 , models.common.SPPF 等
        t = str(m)[8:-2].replace('__main__.', '')   # module 类型
        np = sum(x.numel() for x in m_.parameters())   # 参数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np    # 附加索引，“来自”索引，类型，数字参数
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')   # print

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)    # append to savelist
        layers.append(m_)
        if i == 0:  # 如果是初次迭代，则新创建一个ch（因为形参ch在创建第一个网络模块时需要用到，所以创建网络模块之后再初始化ch）
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)  # 将所有的层封装为nn.Sequential， 对保存的特征图排序


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 1, 2, 3 or cpu')
    parser.add_argument('profile', action='store_true', help='profile model speed')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)    # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # 建立模型
    model = Model(opt.cfg).to(device)
    model.train()

    # profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

