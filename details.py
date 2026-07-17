
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from PIL.ImageFilter import Kernel


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


def compute_conv2d(X, Weight, stride, pad):
    """
    纯数学嵌套循环实现的标准 2D 卷积
    X: (B, C_in, H_in, W_in)
    Weight: (C_out, C_in, Kh, Kw)
    stride: 步长 (假设高宽步长一致)
    pad: 填充大小
    """
    B, C_in, H_in, W_in = X.shape
    C_out, C_in_w, Kh, Kw = Weight.shape

    assert C_in == C_in_w, "输入通道数必须与权重的输入通道数一致"

    # 1. 计算普通卷积的理论输出尺寸
    H_out = (H_in + 2 * pad - Kh) // stride + 1
    W_out = (W_in + 2 * pad - Kw) // stride + 1

    # 初始化输出张量
    Y_manual = torch.zeros((B, C_out, H_out, W_out), device=X.device)

    # 2. 对应论文中的多重求和公式：
    # Y[b, co, i, j] = sum_{ci} sum_{m} sum_{n} X[b, ci, x, y] * W[co, ci, m, n]
    for b in range(B):  # 1. 遍历 Batch 维度
        for co in range(C_out):  # 2. 遍历输出通道 C_out
            for i in range(H_out):  # 3. 遍历输出空间行 H_out
                for j in range(W_out):  # 4. 遍历输出空间列 W_out
                    val = 0.0

                    for ci in range(C_in):  # 5. 遍历输入通道 C_in
                        for m in range(Kh):  # 6. 遍历卷积核高 Kh
                            for n in range(Kw):  # 7. 遍历卷积核宽 Kw

                                # 核心：将输出坐标 (i, j) 配合卷积核内的偏移量 (m, n) 映射回输入张量上的坐标 (x, y)
                                x = i * stride - pad + m
                                y = j * stride - pad + n

                                # 虚拟 Padding 约束：判定映射回的 x, y 是否落在原始有效输入范围内
                                # 如果越界（<0 或 >=H_in/W_in），相当于乘了 Padding 的 0，因此只累加有效范围
                                if 0 <= x < H_in and 0 <= y < W_in:
                                    val += X[b, ci, x, y] * Weight[co, ci, m, n]

                    Y_manual[b, co, i, j] = val

    return Y_manual
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
        # W_tilde=torch.transpose(Weight,2,3)#[H,W,C_out,C_in]
        W_tilde=Weight
    else:
        W_tilde=Weight
    return W_tilde

# def unified_conv(X, W, Tx=1.0, Ty=1.0, Sx=1, Sy=1, pad=0):
#     """统一调度接口"""
#     X_tilde = sample_X(X, Tx, Ty)
#     W_tilde = proc_W(W, Tx, Ty)
#     print("W_tilde.shape", W_tilde.shape)
#     return compute_conv_all(X_tilde, W_tilde, Sx=Sx, Sy=Sy, pad=pad,T=Tx)

def compute_transconv(X,Weight,stride):
    B, C_in, H_in, W_in = X.shape
    C_in, C_out, Kh, Kw = Weight.shape
    # 计算理论输出尺寸
    hout = (H_in - 1) * stride + Kh
    wout = (W_in - 1) * stride + Kw

    # 初始化输出张量
    Y_manual = torch.zeros((1, C_out, hout, wout))

    # 对应论文中的多重求和公式：
    # Y[co, i, j] = sum_{ci} sum_{x in Omega_x} sum_{y in Omega_y} X[ci, x, y] * W[ci, co, m, n]
    for co in range(C_out):  # 遍历输出通道 C_out
        for i in range(hout):  # 遍历输出空间行 H_out
            for j in range(wout):  # 遍历输出空间列 W_out
                val = 0.0
                for ci in range(C_in):  # 遍历输入通道 C_in (公式中的第一个求和符号)
                    for x in range(H_in):  # 遍历输入空间行 H_in (公式中的第二个求和符号)
                        for y in range(W_in):  # 遍历输入空间列 W_in (公式中的第三个求和符号)

                            # 计算当前输入 (x, y) 投射到输出 (i, j) 时，对应的卷积核内部索引 (m, n)
                            m = i  - x * stride
                            n = j  - y * stride

                            # 严格对应约束条件：0 <= m < K_H 且 0 <= n < K_W (即判定 x, y 是否在 Omega 集合中)
                            if 0 <= m < Kh and 0 <= n < Kw:
                                val += X[0, ci, x, y] * Weight[ci, co, m, n]

                Y_manual[0, co, i, j] = val
    return Y_manual