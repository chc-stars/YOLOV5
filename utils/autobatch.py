# -------------------------------
# -*- coding = utf-8 -*-
# @Time : 2023/2/12 13:59
# @Author : chc_stars
# @File : autobatch.py
# @Software : PyCharm
# -------------------------------

from copy import deepcopy

import numpy as np
import torch
from torch.cuda import amp

from utils.general import colorstr
from utils.torch_utils import profile


def check_train_batch_size(model, imgsz=640):
    # 检查yolov5训练的批大小
    with amp.autocast():
        return autobatch(deepcopy(model).train(), imgsz)  # compute optimal bs


def autobatch(model, imgsz=640, fraction=0.9, batch_size=16):
    prefix = colorstr('autobatch； ')
    print(f'{prefix}Computing optimal batch size for --imgsz {imgsz}')
    device = next(model.parameters()).device # get model device
    if device.type == 'cpu':
        print(f'{prefix} CUDA not detected ,using default CPU batch-size {batch_size}')
        return batch_size

    d = str(device).upper()    # CUDA:0
    properties = torch.cuda.get_device_properties(device)  # device properties
    t = properties.total_memory / 1024 ** 3 #  (GiB)
    r = torch.cuda.memory_reserved(device) / 1024 ** 3 # (GiB)
    a = torch.cuda.memory_allocated(device) /1024 ** 3  #(Gib)
    f = t - (r + a)  # free inside reserved
    print(f'{prefix}{d}  ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f} G free')

    batch_sizes = [1, 2, 4, 8, 16]

    try:
        img = [torch.zeros(b, 3, imgsz, imgsz) for b in batch_sizes]
        y = profile(img, model, n=3, device=device)

    except Exception as e:
        print(f'{prefix} {e}')

    y = [x[2] for x in y if x]   # memory[2]
    batch_sizes = batch_size[:len(y)]
    p = np.polyfit(batch_sizes, y, deg=1)   # first degree polynomial fit
    b = int((f * fraction - p[1]) / p[0])   # y intercept (optimal batch size)
    print(f'{prefix} Using batch-size{b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%)')

    return b

