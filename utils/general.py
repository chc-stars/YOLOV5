# -------------------------------
# -*- coding = utf-8 -*-
# @Time : 2023/2/10 19:42
# @Author : chc_stars
# @File : general.py
# @Software : PyCharm
# -------------------------------

import contextlib
import glob
import logging
import math
import os
import platform
import random
import re
import signal
import time
import urllib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml

from utils.download import gsutil_getsize
from utils.metrics import fitness, box_iou

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}.format"})  # format short g, % precision = 5
pd.options.display.max_columns = 10
cv2.setNumThreads(0) # prevent OpenCV multithreading (incompatible with PyTorch DataLoader)
os.environ["NUMEXPR_MAX_THREADS"] = str(min(os.cpu_count(), 8))   # NUMEXpr max threads

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOV5 root direcotory


def set_logging(name=None, verbose=True):
    # Sets level and returns logger
    rank = int(os.getenv("RANk", -1)) # rank in word for  Multi-GPU trainings
    logging.basicConfig(format="%(message)s", level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


LOGGER = set_logging(__name__)  # define globally (used in train.py, val.pu, detect.py , etc.)


class Profile(contextlib.ContextDecorator):
    # Usage: @ Profile() decorator or 'with Profile():'  context manager
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        print(f'Profile results:{time.time() - self.start:.5f}s')


class Timeout(contextlib.ContextDecorator):
    # Usage: @Timeout(seconds) decorator or 'with Timeout(seconds):' context manager
    def __init__(self, seconds, *, timeout_msg='', suppress_timeout_errors=True):
        self.seconds = int(seconds)
        self.timeout_message = timeout_msg
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._timeout_handler) # Set handler for SIGALRM
        signal.alarm(self.seconds)  # start countdown for SIGALRM to be raised

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0) # Cancel SIGALRM if it is scheduled
        if self.suppress and exc_type is TimeoutError:  # Suppress TimeoutError
            return True


class WorkingDirectory(contextlib.ContextDecorator):
    def __init__(self, new_dir):
        self.dir = new_dir  #new dir
        self.cwd = Path.cwd().resolve()      # current dir

    def __enter__(self):       # __enter__ 与 __exit__：输入和退出内置方法，会先执行__enter__方法然后执行__exit__方法。这两个方法通常会用作 上下文管理器使用，比较常见的有 with 、open方法,
        os.chdir(self.dir)          # 被with包裹的代码在执行前先执行__enter__方法，代码结束后会执行__exit__方法

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.cwd)


def try_except(func):
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)

    return handler


def methods(instance):
    # Get class/instance methods
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]


def print_args(name, opt):
    # Print argparser arguments
    LOGGER.info(colorstr(f'{name}: ' + ', '.join(f'{k}={v}' for k, v in vars(opt).items())))


def init_seed(seed=0):
    # 初始化随机数生成器（RNG）种子https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn种子0设置更慢、更可复制，否则更快、更不可复制
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed ==0 else (True, False)


def intersect_dicts(da, db, exclude=()):
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def is_ascii(s=''):
    # 字符串是否由所有ASCII（无UTF）字符组成？（注意python 3.7中引入的str（）.iscii（））
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)


def is_chinese(s='人工智能'):
    # Is string composed of any Chinese characters?
    return re.search('[\u4e00-\u9fff]', s)


def check_yaml(file, suffix=('.yaml', '.yaml')):
    # 搜索/下载YAML文件（如果需要）和返回路径，检查后缀
    return check_file(file, suffix)


def check_file(file, suffix=''):
    # 搜索、下载文件（如果需要） 和返回路径
    check_suffix(file, suffix)  # optional
    file = str(file)  # 转为字符
    if Path(file).is_file() or file == '':   # exists
        return file
    elif file.startswith(('http:/', 'https:/')):   # download
        url = str(Path(file)).replace(':/', '://')  # Path turns ://  ->  :/
        file = Path(urllib.parse.unquote(file).split('?')[0]).name   # '%2F'  to '/', split  https://url.com/file.txt?auth
        if Path(file).is_file():
            print(f'Found {url} locally at {file}')  # 文件已经存在
        else:
            print(f'Downloading {url} to {file}...')
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check

        return file

    else: # search
        files = []
        for d in 'data', 'models', 'utils': # 搜索目录
            files.extend(glob.glob(str(ROOT / d /'**' / file), recursive=True))   # find file
        assert len(files), f'File not found: {file}'  # 断言文件别找到
        assert len(files) == 1, f"multiple files match '{file}',  specify exact path: {files}"  # assert 唯一
        return  files[0]  # 返回文件


def get_latest_run(search_dir="."):
    #  返回/runs中最新“last.pt”的路径（即--恢复自）
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getatime) if last_list else ''


def is_writeable(dir, test=False):
    # Return True if directory has write permissions, test opening a file with write permissions if test=True
    if test:
        file = Path(dir) / 'tmp.txt'

        try:
            with open(file, 'w'):  # 使用写权限打开文件
                pass
            file.unlink()   # 去除文件
            return True
        except OSError:
            return False
    else: # 方法2
        return os.access(dir, os.R_OK)  #


def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


def segments2boxes(segments):
    # 将线段标签转换为方框标签, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape(n, 2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw # top left x
    y[:, 1] = h * x[:, 1] + padh # top left y
    return y


def file_size(path):
    # 返回文件、或目录大小（MB）
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1E6
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / 1E6
    else:
        return 0.0

def check_online():
    # 检查网络链接
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # 检查主机可访问性
        return True
    except OSError:
        return False


def check_python(minimum = '3.6.2'):
    # 检查当前python版本以及与需求版本对比
    check_version(platform.python_version(), minimum, name="Python ", hard=True)


def check_version(current='0.0.0', minimum='0.0.0', name="version", pinned=False, hard=False):
    # 检查版本和需求版本
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum) # bool
    if hard: #
        assert result, f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'
    else:
        return result


def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1):
    # Multi-threaded file download and unzip function, used in data.yaml for autodownload
    def download_one(url, dir):
        # Download 1 file
        f = dir / Path(url).name  # filename
        if Path(url).is_file():  # exists in current path
            Path(url).rename(f)  # move to dir
        elif not f.exists():
            print(f'Downloading {url} to {f}...')
            if curl:
                os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")  # curl download, retry and resume on fail
            else:
                torch.hub.download_url_to_file(url, f, progress=True)  # torch download
        if unzip and f.suffix in ('.zip', '.gz'):
            print(f'Unzipping {f}...')
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)  # unzip
            elif f.suffix == '.gz':
                os.system(f'tar xfz {f} --directory {f.parent}')  # unzip
            if delete:
                f.unlink()  # remove zip

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multi-threaded
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


def check_dataset(data, autodownload=True):
    # Download and/or unzip dataset if not found locally
    # Usage: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip

    # Download (optional)
    extract_dir = ''
    if isinstance(data, (str, Path)) and str(data).endswith('.zip'):  # i.e. gs://bucket/dir/coco128.zip
        download(data, dir='../datasets', unzip=True, delete=False, curl=False, threads=1)
        data = next((Path('../datasets') / Path(data).stem).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        with open(data, errors='ignore') as f:
            data = yaml.safe_load(f)  # dictionary

    # Parse yaml
    path = extract_dir or Path(data.get('path') or '')  # optional 'path' default to '.'
    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]

    assert 'nc' in data, "Dataset 'nc' key missing."
    if 'names' not in data:
        data['names'] = [f'class{i}' for i in range(data['nc'])]  # assign class names if missing
    train, val, test, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            if s and autodownload:  # download script
                root = path.parent if 'path' in data else '..'  # unzip directory i.e. '../'
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    f = Path(s).name  # filename
                    print(f'Downloading {s} to {f}...')
                    torch.hub.download_url_to_file(s, f)
                    Path(root).mkdir(parents=True, exist_ok=True)  # create root
                    ZipFile(f).extractall(path=root)  # unzip
                    Path(f).unlink()  # remove zip
                    r = None  # success
                elif s.startswith('bash '):  # bash script
                    print(f'Running {s} ...')
                    r = os.system(s)
                else:  # python script
                    r = exec(s, {'yaml': data})  # return None
                print(f"Dataset autodownload {f'success, saved to {root}' if r in (0, None) else 'failure'}\n")
            else:
                raise Exception('Dataset not found.')

    return data  # dictionary


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


@try_except
@WorkingDirectory(ROOT)
def check_git_status():
    # Recommend 'git pull' if code is out of date  # 如果代码过期， 建议使用‘git pull
    msg = ', for updates see https://github.com/ultralytics/yolov5'
    print(colorstr('github: '), end='')
    assert Path('.git').exists(), 'skipping check (not a git repository)' + msg
    assert not is_docker(), 'skipping check (Docker image)' + msg
    assert check_online(), 'skipping check (offline)' + msg

    cmd = 'git fetch && git config --get remote.origin.url'
    url = check_output(cmd, shell=True, timeout=5).decode().strip().rstrip('.git')  # git fetch
    branch = check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()  # checked out
    n = int(check_output(f'git rev-list {branch}..origin/master --count', shell=True))  # commits behind
    if n > 0:
        s = f"⚠️ YOLOv5 is out of date by {n} commit{'s' * (n > 1)}. Use `git pull` or `git clone {url}` to update."
    else:
        s = f'up to date with {url} ✅'
    print(emojis(s))  # emoji-safe


def is_docker():
    # Is environment a Docker container?
    return Path('/workspace').exists()  # or Path('/.dockerenv').exists()


@try_except
def check_requirements(requirements=ROOT / 'requirements.txt', exclude=(), install=True):
    # Check installed dependencies meet requirements (pass *.txt file or list of packages)
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()  # check python version
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)
        assert file.exists(), f"{prefix} {file.resolve()} not found, check failed."
        requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(file.open()) if x.name not in exclude]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # number of packages updates
    for r in requirements:
        try:
            pkg.require(r)
        except Exception as e:  # DistributionNotFound or VersionConflict if requirements not met
            s = f"{prefix} {r} not found and is required by YOLOv5"
            if install:
                print(f"{s}, attempting auto-update...")
                try:
                    assert check_online(), f"'pip install {r}' skipped (offline)"
                    print(check_output(f"pip install '{r}'", shell=True).decode())
                    n += 1
                except Exception as e:
                    print(f'{prefix} {e}')
            else:
                print(f'{s}. Please install and rerun your command.')

    if n:  # if packages updated
        source = file.resolve() if 'file' in locals() else requirements
        s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        print(emojis(s))


def check_img_size(imgsz, s=32, floor=0):
    #  验证图像大小是否为每个维度的步长倍数
    if isinstance(imgsz, int): # 整数， i.e  img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)

    else: # list i.e img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]

    if new_size != imgsz:
        print(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')

    return new_size


def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    # 检查文件是否为可支持的后缀
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # 文件后缀
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def url2file(url):
    #  Convert URL to  filename, i.e
    url = str(Path(url)).replace(':/', '://')
    file = Path(urllib.parse.unquote(url)).name.split('?')[0]  #
    return file


def make_divisible(x, divisor):
    # 返回可被除数整除的书
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2 ) * (y2 - y1) + y1


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)   # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)      # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # 基于类别权重和图像内容生成图像权重
    class_counts = np.array([np.bincount(x[:, 0].astype(np.int), minlength=nc) for x in labels])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)

    return image_weights


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounind boxes to image shape (height, weidth)
    # 将边界xyxy边界框剪裁为图像形状（高度、宽度）
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2

    else:  # np.array(faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # 将坐标（xyxy）从img1_shape重缩放为img0_shape
    if ratio_pad is None:   # 根据img0_shape计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1]/ img0_shape[0])   # gain = old /new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]   # x padding
    coords[:, [1, 3]] -= pad[1]   # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1 = top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2]/2   # top left x
    y[:, 1] = x[:, 1] - x[:, 3]/2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2]/2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3]/2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=.25, iou_thres=.45, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

      Returns:
           list of detections, on (n,6) tensor per image [xyxy, conf, cls]
      """

    nc = prediction.shape[2] -5  # 类别数量
    xc = prediction[..., 4] > conf_thres  # 候选框

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096 # （像素）最小和最大框宽度和高度
    max_nms = 30000

    time_limit = 10.0
    redundant = True  #  require redundant detections

    multi_label &= nc > 1  # multiple labels per box (adds. 0.5ms/ img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    for xi, x in enumerate(prediction):   # image index, iamge inference
        # Apply constraints
        x = x[xc[xi]]   # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[: 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:.0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # if none remain process next image
        if not x.shape[0]:
            continue

        # Conpute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf ,cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j+5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[: 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check  shape
        n = x.shape[0]   # num of boxes
        if not n:  # no box
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]   # sort by confidence

        c = x[:, 5:6] * (0 if agnostic else max_wh) # classes
        boxes, scores = x[:, :4] + c, x[:, 4] # boxes (offset by classes), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)   #NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):   # Merge NMS (boxes merged using weight mean)
            # update boxes as boxes(i, 4) - weights(i, n) * boxes(n, 4)
            iou  = box_iou(boxes[i], boxes) > iou_thres # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1)  > 1]  # require redundancy

        output[xi] = x[i]
        if(time.time() - t) > time_limit:
            print(f'Warning:  NMS time limit {time_limit}s exceeded')
            break   # time limit exceeded

    return output


def strip_optimizer(f='best.pt', s=''): # from utils.general import *; strip_optimizer()
    # 将优化器从“f”中剥离以完成训练，也可以另存为“s”
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']   # #用ema替换模型

    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates': # keys
        x[k] = None

    x['epoch'] = -1
    x['model'].half()   # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print(f"Optimizer stripped from {f}, {('saved as %s, '  %s) if s else ''} {mb:.1f}MB")


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def print_mutation(results, hyp, save_dir, bucket):
    evolve_csv, results_csv, evolve_yaml = save_dir / 'evolve.csv', save_dir / 'results.csv', save_dir / 'hyp_evolve.yaml'
    keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
            'val/box_loss', 'val/obj_loss', 'val/cls_loss') + tuple(hyp.keys())  # [results + hyps]
    keys = tuple(x.strip() for x in keys)
    vals = results + tuple(hyp.values())
    n = len(keys)

    # Download (optional)
    if bucket:
        url = f'gs://{bucket}/evolve.csv'
        if gsutil_getsize(url) > (os.path.getsize(evolve_csv) if os.path.exists(evolve_csv) else 0):
            os.system(f'gsutil cp {url} {save_dir}')  # download evolve.csv if larger than local

    # Log to evolve.csv
    s = '' if evolve_csv.exists() else (('%20s,' * n % keys).rstrip(',') + '\n')  # add header
    with open(evolve_csv, 'a') as f:
        f.write(s + ('%20.5g,' * n % vals).rstrip(',') + '\n')

    # Print to screen
    print(colorstr('evolve: ') + ', '.join(f'{x.strip():>20s}' for x in keys))
    print(colorstr('evolve: ') + ', '.join(f'{x:20.5g}' for x in vals), end='\n\n\n')

    # Save yaml
    with open(evolve_yaml, 'w') as f:
        data = pd.read_csv(evolve_csv)
        data = data.rename(columns=lambda x: x.strip())  # strip keys
        i = np.argmax(fitness(data.values[:, :7]))  #
        f.write('# YOLOv5 Hyperparameter Evolution Results\n' +
                f'# Best generation: {i}\n' +
                f'# Last generation: {len(data)}\n' +
                '# ' + ', '.join(f'{x.strip():>20s}' for x in keys[:7]) + '\n' +
                '# ' + ', '.join(f'{x:>20.5g}' for x in data.values[i, :7]) + '\n\n')
        yaml.safe_dump(hyp, f, sort_keys=False)

    if bucket:
        os.system(f'gsutil cp {evolve_csv} {evolve_yaml} gs://{bucket}')  # upload


def user_config_dir(dir='Ultralytics', env_var='YOLOV5_CONFIG_DIR'):
    # 用户配置目录的返回路径。如果存在，首选环境变量。如果需要，制作目录。
    env = os.getenv(env_var)
    if env:
        path = Path(env)  # use environment variable
    else:
        cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', "Darwin": 'Library/Application Support'}  # OS dirs
        path = Path.home() / cfg.get(platform.system(), '')  # OS-specific config dir
        path = (path if is_writeable(path) else Path('/tmp')) / dir  # GCP and AWS lambda fix, only /tmp is writeable
    path.mkdir(exist_ok=True)   # make if required
    return path


def apply_classifier(x, model, img, im0):
    # Apply a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # reshape and pad cutouts
            b = xyxy2xywh(d[:, :4]) # boxes
            b[:, 2:] = b[:, 2:] .max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30 # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]): int(a[2])]
                im = cv2.resize(cutout, (224, 224))   # BGR

                im = im[:, :, :: -1].transpose(2, 0, 1) # BGR to RGB, to 3 * 416 *416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float 32
                im /= 255  # 0-255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2 ]  # retain matching class detection

    return x


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def is_colab():
    # Is environment a Google Colab instance?
    try:
        import google.colab
        return True
    except ImportError:
        return False


def check_imshow():
    # Check if environment supports image displays
    try:
        assert not is_docker(), 'cv2.imshow() is disabled in Docker environments'
        assert not is_colab(), 'cv2.imshow() is disabled in Google Colab environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False


if __name__ == '__main__':
    LOGGER.info("nihao, j ")
    print(file_size("E:\mians/面试总结.docx"))
    # scale_coords([400, 600], (200, 200), [600, 600])
    print(check_online())
