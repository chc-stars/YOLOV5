# -------------------------------
# -*- coding = utf-8 -*-
# @Time : 2023/2/13 10:26
# @Author : chc_stars
# @File : resume.py
# @Software : PyCharm
# -------------------------------

#  在yolov5/dir恢复所有中断的培训，包括DDP培训

import os
import sys
from pathlib import Path

import torch
import yaml

FILE = Path(__file__).resolve()  # E:\CODE\CV\Object_detection\YOLO\YOLOv5\Our_YOLO5\utils\aws\resume.py
ROOT = FILE.parents[2]   #  YOLOV5 根目录  E:\CODE\CV\Object_detection\YOLO\YOLOv5\Our_YOLO5
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))   # 添加ROOT 到 PATH

port = 0  #  --master_port
path = Path('').resolve()
for last in path.rglob("*/**/last.pt"):
    ckpt = torch.load(last)
    if ckpt['optimizer'] is None:
        continue

    # Load opt.yaml
    with open(last.parent.parent / 'opt.yml', errors='ignore') as f:
        opt = yaml.safe_load(f)

    #  Get device count
    d = opt['device'].split(',')  # devices
    nd = len(d)  # number of devices
    ddp = nd > 1 or (nd == 0 and torch.cuda.device_count()  > 1)  # distribured data parallel

    if ddp:   # 多GPU
        port += 1
        cmd = f'python -m torch.distrbuted.run --nproc_per_node {nd} --master_port {port} train.py --resume {last}'
    else:  # 单GPU
        cmd = f'python train.py --resume{last}'

    cmd += ' > / dev / null 2 >&1 &'  # redirect output to dev/null and run in daemon thread
    print(cmd)
    os.system(cmd)

