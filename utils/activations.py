# -------------------------------
# -*- coding = utf-8 -*-
# @Time : 2023/2/10 18:13
# @Author : chc_stars
# @File : activations.py
# @Software : PyCharm
# -------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class SILU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):
    @staticmethod
    def forward(x):
        # return x * F.hardsigoid(x) # for torchscript and CoreML
        return x * F.hardswish(x+3, 0.0, 6.0) / 6.0 # for torchscript, CoreML and ONNX


class MiSh(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output + (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)
