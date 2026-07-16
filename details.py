
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn

# def compute_conv(X, Weight, x, y, c2, Sx, Sy):
#
#     B,H,W,C_in=X.shape
#     Kh,Kw,C_in,C_out=Weight.shape
#
#     y=0
#     for c1 in range(C_in):
#         for j in range(Kh):
#             for i in range(Kw):
#                 y+=X[x*Sx+i-1,y*Sy+j-1,c1]
#     return y


def compute_conv_pixel(X, Weight, x, y, c2, Sx, Sy):
    """
    计算单通道 c2，当前空间坐标 (x, y) 的卷积结果
    X 形状: (B, C_in, H, W)
    Weight 形状: (C_out, C_in, Kh, Kw)
    """
    # 1. 计算感受野切片坐标（已经做过物理 pad，不要减 1）
    start_h = x * Sx
    end_h = start_h + Weight.shape[2]  # Kh

    start_w = y * Sy
    # 修复你原代码中统一用 Sy 的问题：高用 Sx（步长x），宽用 Sy（步长y）
    end_w = start_w + Weight.shape[3]  # Kw

    # 2. 核心：切片提取局部窗口 (Patch)
    # 形状为: (B, C_in, Kh, Kw)
    patch = X[:, :, start_h:end_h, start_w:end_w]

    # 3. 取出对应的卷积核权重
    # Weight[c2] 的形状是 (C_in, Kh, Kw)
    # 加上 [None, ...] 变成 (1, C_in, Kh, Kw)，方便与具有 Batch 维度的 patch 进行广播乘法
    kernel = Weight[None, c2, :, :, :]

    # 4. 逐元素相乘并求和
    # patch * kernel 形状仍为 (B, C_in, Kh, Kw)
    # 为了保留 Batch 维度（即 Y_out[:, c2, x, y] 需要接收一个长度为 B 的向量）
    # 我们只对通道维和空间维求和（dim=[1, 2, 3]），保留第 0 维 (Batch)
    # print("patch.shape", patch.shape)
    # print("kernel.shape", kernel.shape)
    out_val = torch.sum(patch * kernel, dim=[0,1, 2, 3])

    return out_val

def compute_conv_all(X_tilde, W_tilde, Sx, Sy, pad, T):
    # 1. 获取未 Padding 前的原始尺寸
    B, C_in, H_in, W_in = X_tilde.shape
    C_out, C_in_w, Kh, Kw = W_tilde.shape

    assert C_in == C_in_w, "输入张量的通道数必须与权重的输入通道数一致"

    # 2. 根据是卷积还是反卷积，计算不同的输出尺寸
    if T<1:
        # 反卷积 (Transposed Convolution) 尺寸公式
        # 假设 dilation=1, output_padding=0
        H_out = (H_in - 1) * Sx - 2 * pad + Kh
        W_out = (W_in - 1) * Sy - 2 * pad + Kw
    else:
        # 普通卷积 (Convolution) 尺寸公式
        H_out = (H_in + 2 * pad - Kh) // Sx + 1
        W_out = (W_in + 2 * pad - Kw) // Sy + 1

    # 3. 执行物理 Padding
    # 对于 BCHW，(pad, pad, pad, pad) 正好填充最后两个空间维度 H 和 W
    X_padded = F.pad(X_tilde, (pad, pad, pad, pad))

    # 输出形状为 (B, C_out, H_out, W_out)
    Y_out = torch.zeros((B, C_out, H_out, W_out), device=X_tilde.device)

    for x in range(H_out):
        for y in range(W_out):
            for c2 in range(C_out):
                # 注意：传入的是经过 pad 处理后的 X_padded
                Y_out[:, c2, x, y] = compute_conv_pixel(X_padded, W_tilde, x, y, c2, Sx, Sy)

    return Y_out
def sample_X(X,Tx,Ty):

    B,C_in,H,W=X.shape
    # 当 Tx, Ty < 1 时 (Expansion / 转置卷积)：插入 0
    # todo 目前只有T，没有区别化Tx Ty
    if Tx < 1 and Ty < 1:
        Sx_expand = int(1 / Tx)
        Sy_expand = int(1 / Ty)

        # 计算插零后的新尺寸：(原始尺寸 - 1) * 膨胀率 + 1
        H_tilde = H  * Sy_expand
        W_tilde = W  * Sx_expand

        # 创建全 0 张量，并将 X 的值填入对应整数坐标处 (对应公式中 x*Tx 为整数的条件)
        X_tilde = torch.zeros(B, C_in, H_tilde, W_tilde, dtype=X.dtype, device=X.device)
        X_tilde[:, :, ::Sy_expand, ::Sx_expand] = X
    elif Tx > 1 and Ty > 1:# 要删掉，错误的
        Tx_int = int(Tx)
        Ty_int = int(Ty)
        X_tilde = X[:, :, ::Ty_int, ::Tx_int]
    else:
        X_tilde = X
    return X_tilde

def proc_W(Weight, Tx, Ty):
    if Tx<1 and Ty<1:
        # W_tilde = torch.rot90(Weight, k=2, dims=[2, 3])# W 180翻转
        # W_tilde=torch.transpose(Weight,2,3)#[H,W,Cout,Cin]
        W_tilde=Weight
    else:
        W_tilde=Weight
    return W_tilde

def unified_conv(X, W, Tx=1.0, Ty=1.0, Sx=1, Sy=1, pad=0):
    """统一调度接口"""
    X_tilde = sample_X(X, Tx, Ty)
    W_tilde = proc_W(W, Tx, Ty)
    print("W_tilde.shape", W_tilde.shape)
    return compute_conv_all(X_tilde, W_tilde, Sx=Sx, Sy=Sy, pad=pad,T=Tx)