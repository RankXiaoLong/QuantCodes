# import datetime
# import math
# import time
# from datetime import timedelta
# import pandas as pd
# import statsmodels.api as sm
# import matplotlib
# from matplotlib import pyplot as plt
# from tqdm import tqdm
#
# from torch.utils.data import DataLoader
# from torch import optim
# import torch.nn.functional as F
# import tushare as ts

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os


# inception1:convolution:10,pooling:3
class Inception1(nn.Module):
    # 该部分主要提供窗口为10的卷积，以及尺寸为3的池化
    # 输入为9*30，输出为513*1
    # 我们不需要激活函数，因为我们希望我们的特征提取层有和wq101因子类似的构造逻辑
    def __init__(self, num: list, num_rev: list, stride: int):
        super(Inception1, self).__init__()
        self.num = num
        self.num_rev = num_rev
        self.stride = stride
        self.bc1 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc2 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc3 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc4 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc5 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc6 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc7 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc8 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc9 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc10 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bcGRU = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc_skew = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc_kurt = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc_rn = nn.BatchNorm2d(1, eps=1e-5, affine=True)

        self.bc_pool1 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc_pool2 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc_pool3 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.min_pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))  # 实际上没有最小池化

        self.residual = nn.Conv2d(1, 18, kernel_size=(1, 1), stride=(1, 10))
        self.gru = nn.GRU(375, 30, 2)  # 两层的GRU
        self.bidirectional = False  # no Bi-GRU

    def forward(self, data1, index:int, is_trian:bool):  # B*1*9*30
        # data1 = data1.detach().numpy()
        num = self.num
        num_rev = self.num_rev
        conv1 = self.ts_cov4d(data1, num, self.stride).to(torch.float)
        conv4 = self.ts_corr4d(data1, num, self.stride)
        conv6 = self.ts_decaylinear(data1, self.stride).to(torch.float)
        conv7 = self.ts_std(data1, self.stride).to(torch.float)
        conv_mean = self.ts_mean(data1, self.stride).to(torch.float)
        conv_zcore= self.ts_zcore(data1, self.stride).to(torch.float)
        conv10 = self.ts_mul(data1, num, num_rev, self.stride).to(torch.float)
        conv_skew = self.ts_skew(data1, self.stride).to(torch.float)
        # conv_kurt = self.ts_kurt(data1, self.stride).to(torch.float)
        conv_rn = self.ts_return(data1, self.stride).to(torch.float)

        batch1 = self.bc1(conv1)
        batch4 = self.bc4(conv4)
        batch6 = self.bc6(conv6)
        batch7 = self.bc7(conv7)
        batch_mean = self.bc8(conv_mean)
        batch_zcore = self.bc9(conv_zcore)
        batch10 = self.bc10(conv10)

        # # 输入GRU，出现梯度爆炸
        # batch_skew = self.bc_skew(conv_skew)
        # # batch_kurt = self.bc_kurt(conv_kurt)
        # batch_rn = self.bc_rn(conv_rn)

        # # 算子因子保存
        # batch4_corrCT = batch4[:, :, 39, :]
        # batch4_corrVT = batch4[:, :, 94, :]
        # if is_trian:
        #     np.save(f'E:\【Intern】\AlphaNet\zz800SingleFactors\Incenption1_E{index}.npy', batch4_corrCT.cpu().detach().numpy())
        #     np.save(f'E:\【Intern】\AlphaNet\zz800SingleFactors\Incenption1_E{index}.npy', batch4_corrVT.cpu().detach().numpy())
        # else:
        #     np.save(f'E:\【Intern】\AlphaNet\zz800SingleFactors\Incenption1_test.npy', batch4_corrCT.cpu().detach().numpy())
        #     np.save(f'E:\【Intern】\AlphaNet\zz800SingleFactors\Incenption1_test.npy', batch4_corrVT.cpu().detach().numpy())

        concat = [batch1, batch4, batch6, batch7, batch_mean, batch_zcore, batch10] # batch_skew, batch_rn
        feature1 = torch.cat(concat, axis=2)
        data = feature1.squeeze(1).transpose(1, 0).transpose(2, 0)  # N*1*9*30 -> 30*100*9
        output, hn = self.gru(data)
        h = hn[-(1 + int(self.bidirectional)):]
        x = torch.cat(h.split(1), dim=-1).squeeze(0)
        output_inp1 = self.bcGRU(x.reshape([1, 1, x.shape[0], x.shape[1]])).reshape([-1, 30])
        return output_inp1

    def ts_zcore(self, Matrix:torch.tensor, stride: int):
        W, H = Matrix.shape[3], Matrix.shape[2]
        if W % stride == 0:
            Index_list = list(np.arange(0, W + stride, stride))
        else:
            mod = W % stride
            Index_list = list(np.arange(0, W + stride - mod, stride)) + [W]
        l = []
        for i in range(len(Index_list) - 1):
            start, end = Index_list[i], Index_list[i + 1]
            data = Matrix[:, :, :, start:end]
            mean, std = data.mean(axis=3, keepdims=True), data.std(axis=3, keepdims=True) + 0.01
            l.append(mean / std)
        tensor_l = torch.stack([i for i in l], 0)
        return torch.squeeze(tensor_l).permute(1, 2, 0).reshape(-1, 1, H, len(Index_list) - 1)

    def ts_cov4d(self, Matrix, num: list, stride: int):
        W, H = Matrix.shape[3], Matrix.shape[2]
        new_H = len(num)
        if W % stride == 0:
            Index_list = list(np.arange(0, W + stride, stride))
        else:
            mod = W % stride
            Index_list = list(np.arange(0, W + stride - mod, stride)) + [W]
        l = []# 存放长度为num的协方差
        for i in range(len(Index_list) - 1):
            start_index, end_index = Index_list[i], Index_list[i + 1]
            data1 = Matrix[:, :, num, start_index:end_index]
            mean1 = data1.mean(axis=4, keepdims=True)
            spread1 = data1 - mean1
            cov = ((spread1[:, :, :, 0, :] * spread1[:, :, :, 1, :]).sum(axis=3, keepdims=True)
                   / (data1.shape[4] - 1)).mean(axis=3, keepdims=True)
            # l.append(cov)  # len(num) * N * 2
            l.append(cov)
        tensor_l = torch.stack([i for i in l], 0)
        # cov = torch.squeeze(tensor_l).permute(1, 2, 0).reshape(-1, 1, new_H, len(Index_list) - 1)
        # cov = torch.squeeze(tensor_l).transpose(1, 2, 0).reshape(-1, 1, new_H, len(Index_list) - 1)
        return torch.squeeze(tensor_l).permute(1, 2, 0).reshape(-1, 1, new_H, len(Index_list) - 1) # torch.from_numpy(cov)

    def ts_corr4d(self, Matrix, num, stride):
        W = Matrix.shape[3]
        H = Matrix.shape[2]
        new_H = len(num)
        if W % stride == 0:
            Index_list = list(np.arange(0, W + stride, stride))
        else:
            mod = W % stride
            Index_list = list(np.arange(0, W + stride - mod, stride)) + [W]
        l = []  # 存放长度为num的相关系数
        for i in range(len(Index_list) - 1):
            start = Index_list[i]
            end = Index_list[i + 1]
            data1 = Matrix[:, :, num, start:end]

            std1 = data1.std(axis=4, keepdims=True)
            std = std1[:, :, :, 0, :] * std1[:, :, :, 1, :]
            l.append(std)
        fct = (data1.shape[4] - 1) / data1.shape[4]
        # tensor_l = torch.tensor([item.cpu().detach().numpy() for item in l]).cuda()
        tensor_l = torch.stack([i for i in l], 0)
        # std = np.squeeze(np.array(l)).transpose(1, 2, 0).reshape(-1, 1, new_H, len(Index_list) - 1) + 0.01
        std = torch.squeeze(tensor_l).permute(1, 2, 0).reshape(-1, 1, new_H, len(Index_list) - 1) + 0.01
        cov = self.ts_cov4d(Matrix, num, stride)
        return (cov / std) * fct

    def ts_max(self, Matrix, stride):
        W, H = Matrix.shape[3], Matrix.shape[2]
        if W % stride == 0:
            Index_list = list(np.arange(0, W + stride, stride))
        else:
            mod = W % stride
            Index_list = list(np.arange(0, W + stride - mod, stride)) + [W]
        l = []
        for i in range(len(Index_list) - 1):
            start_index, end_index = Index_list[i], Index_list[i + 1]
            data1 = Matrix[:, :, :, start_index:end_index]
            l.append(data1.min(axis=3))
        # ts_max = np.squeeze(np.array(l)).transpose(1, 2, 0).reshape(-1, 1, H, len(Index_list) - 1)
        tensor_l = torch.stack([i for i in l], 0)
        return torch.squeeze(tensor_l).permute(1, 2, 0).reshape(-1, 1, H, len(Index_list) - 1) #  torch.from_numpy(ts_max)

    def ts_min(self, Matrix, stride):
        W, H = Matrix.shape[3], Matrix.shape[2]
        if W % stride == 0:
            Index_list = list(np.arange(0, W + stride, stride))
        else:
            mod = W % stride
            Index_list = list(np.arange(0, W + stride - mod, stride)) + [W]
        l = []
        for i in range(len(Index_list) - 1):
            start_index, end_index = Index_list[i], Index_list[i + 1]
            data1 = Matrix[:, :, :, start_index:end_index]
            l.append(data1.min(axis=3))
        tensor_l = torch.stack([i for i in l], 0)
        # ts_min = np.squeeze(np.array(l)).transpose(1, 2, 0).reshape(-1, 1, H, len(Index_list) - 1)
        return torch.squeeze(tensor_l).permute(1, 2, 0).reshape(-1, 1, H, len(Index_list) - 1) # torch.from_numpy(ts_min)

    def ts_return(self, Matrix, stride):
        W, H = Matrix.shape[3], Matrix.shape[2]
        new_H = H
        if W % stride == 0:
            Index_list = list(np.arange(0, W + stride, stride))
        else:
            mod = W % stride
            Index_list = list(np.arange(0, W + stride - mod, stride)) + [W]
        l = []
        for i in range(len(Index_list) - 1):
            start_index, end_index = Index_list[i], Index_list[i + 1]
            data1 = Matrix[:, :, :, start_index:end_index]
            return_ = data1[:, :, :, -1] / (data1[:, :, :, 0] + 0.1) - 1
            l.append(return_)
        tensor_l = torch.stack([i for i in l], 0)
        return torch.squeeze(tensor_l).permute(1, 2, 0).reshape(-1, 1, H, len(Index_list) - 1)

    def ts_decaylinear(self, Matrix, stride):
        W = Matrix.shape[3]
        H = Matrix.shape[2]
        new_H = H
        if W % stride == 0:
            Index_list = list(np.arange(0, W + stride, stride))
        else:
            mod = W % stride
            Index_list = list(np.arange(0, W + stride - mod, stride)) + [W]
        l = []
        for i in range(len(Index_list) - 1):
            start, end = Index_list[i], Index_list[i + 1]
            range_ = end - start
            weight = torch.arange(1, range_ + 1)
            weight = weight / weight.sum()  # 权重向量
            weight = weight.to(Matrix.device)
            data = Matrix[:, :, :, start:end]
            # wd = (data * weight).sum(axis=3, keepdims=True)
            l.append((data * weight).sum(axis=3, keepdims=True))
        tensor_l = torch.stack([i for i in l], 0)
        # weight_decay = np.squeeze(np.array(l)).transpose(1, 2, 0).reshape(-1, 1, new_H, len(Index_list) - 1)
        return torch.squeeze(tensor_l).permute(1, 2, 0).reshape(-1, 1, new_H, len(Index_list) - 1)

    def ts_std(self, Matrix, stride):
        W, H = Matrix.shape[3], Matrix.shape[2]
        if W % stride == 0:
            Index_list = list(np.arange(0, W + stride, stride))
        else:
            mod = W % stride
            Index_list = list(np.arange(0, W + stride - mod, stride)) + [W]
        l = []
        for i in range(len(Index_list) - 1):
            start, end = Index_list[i], Index_list[i + 1]
            data = Matrix[:, :, :, start:end]
            l.append(data.std(axis=3, keepdims=True))
        tensor_l = torch.stack([i for i in l], 0)
        return torch.squeeze(tensor_l).permute(1, 2, 0).reshape(-1, 1, H, len(Index_list) - 1)

    def ts_mean(self, Matrix, stride):
        W, H = Matrix.shape[3], Matrix.shape[2]
        # new_H = H
        if W % stride == 0:
            Index_list = list(np.arange(0, W + stride, stride))
        else:
            mod = W % stride
            Index_list = list(np.arange(0, W + stride - mod, stride)) + [W]
        l = []
        for i in range(len(Index_list) - 1):
            start, end = Index_list[i], Index_list[i + 1]
            data = Matrix[:, :, :, start:end]
            l.append(data.mean(axis=3, keepdims=True))
        tensor_l = torch.stack([i for i in l], 0)
        return torch.squeeze(tensor_l).permute(1, 2, 0).reshape(-1, 1, H, len(Index_list) - 1)

    def ts_return(self, Matrix, stride):
        W, H = Matrix.shape[3], Matrix.shape[2]
        # new_H = H
        if W % stride == 0:
            Index_list = list(np.arange(0, W + stride, stride))
        else:
            mod = W % stride
            Index_list = list(np.arange(0, W + stride - mod, stride)) + [W]
        l = []
        for i in range(len(Index_list) - 1):
            start, end = Index_list[i], Index_list[i + 1]
            data = Matrix[:, :, :, start:end]
            l.append((data[:, :, :, -1] / (data[:, :, :, 0]+0.01) - 1))
        tensor_l = torch.stack([i for i in l], 0)
        return torch.squeeze(tensor_l).permute(1, 2, 0).reshape(-1, 1, H, len(Index_list) - 1)

    def ts_kurt(self, Matrix, stride):
        W, H = Matrix.shape[3], Matrix.shape[2]
        # new_H = H
        if W % stride == 0:
            Index_list = list(np.arange(0, W + stride, stride))
        else:
            mod = W % stride
            Index_list = list(np.arange(0, W + stride - mod, stride)) + [W]
        l = []
        for i in range(len(Index_list) - 1):
            start, end = Index_list[i], Index_list[i + 1]
            data = Matrix[:, :, :, start:end]
            data_var = data.var(axis=3, keepdim=True)  # 计算方差
            data_kurt = ((data - data_var) ** 4).mean(axis=3, keepdims=True) / pow(data_var, 2) / 1000
            l.append(data_kurt)
        tensor_l = torch.stack([i for i in l], 0)
        return torch.squeeze(tensor_l).permute(1, 2, 0).reshape(-1, 1, H, len(Index_list) - 1)

    def ts_skew(self, Matrix, stride):
        W, H = Matrix.shape[3], Matrix.shape[2]
        # new_H = H
        if W % stride == 0:
            Index_list = list(np.arange(0, W + stride, stride))
        else:
            mod = W % stride
            Index_list = list(np.arange(0, W + stride - mod, stride)) + [W]
        l = []
        for i in range(len(Index_list) - 1):
            start, end = Index_list[i], Index_list[i + 1]
            data = Matrix[:, :, :, start:end]
            data_mean = data.mean(axis=3, keepdims=True)
            data_skew = ((data - data_mean) ** 3).mean(axis=3, keepdim=True)
            l.append(data_skew)
        tensor_l = torch.stack([i for i in l], 0)
        return torch.squeeze(tensor_l).permute(1, 2, 0).reshape(-1, 1, H, len(Index_list) - 1)

    def ts_mul(self, Matrix, num, num_rev, stride):
        W, H = Matrix.shape[3], Matrix.shape[2]
        new_H = len(num)
        if W % stride == 0:
            Index_list = list(np.arange(0, W + stride, stride))
        else:
            mod = W % stride
            Index_list = list(np.arange(0, W + stride - mod, stride)) + [W]
        l = []
        # 思路来源：换手率*收益率序列的均值
        for i in range(len(Index_list) - 1):
            start, end = Index_list[i], Index_list[i + 1]
            data1, data2 = Matrix[:, :, num, start:end], Matrix[:, :, num_rev, start:end]
            mul = (data1 * data2).mean(axis=4, keepdims=True).mean(axis=3, keepdims=True)
            l.append(mul)
        tensor_l = torch.stack([i for i in l], 0)
        return torch.squeeze(tensor_l).permute(1, 2, 0).reshape(-1, 1, new_H, len(Index_list) - 1)


# inception2:convolution:3,pooling:10
class Inception2(Inception1):
    """
    继承 Inception1: 修改卷积核参数 和 池化参数；其他属性基本不变化
    --------------------------------------------------------------
    该部分主要提供窗口为3的卷积，以及尺寸为10的池化
    输入为9*30，输出为513*1
    我们不需要激活函数，因为我们希望我们的特征提取层有和wq101因子类似的构造逻辑
    """
    def __init__(self, num, num_rev, stride):
        super(Inception2, self).__init__(num, num_rev, stride)
        self.num = num
        self.num_rev = num_rev
        self.stride = stride
        self.bc1 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc2 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc3 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc4 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc5 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc6 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc7 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc8 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc9 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc10 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc_skew = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc_kurt = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc_rn = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        # self.bc10 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bcGRU = nn.BatchNorm2d(1, eps=1e-5, affine=True)

        self.bc_pool1 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc_pool2 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.bc_pool3 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 10), stride=(1, 10))
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 10), stride=(1, 10))
        self.min_pool = nn.MaxPool2d(kernel_size=(1, 10), stride=(1, 10))  # 实际上没有最小池化
        self.residual = nn.Conv2d(1, 18, kernel_size=(1, 1), stride=(1, 10))
        self.gru = nn.GRU(375, 30, 2)  # 两层的GRU, 375 是feature的维度
        self.bidirectional = False  # no Bi-GRU

    def forward(self, data1, index, is_trian):
        """ B*1*9*30 """
        # data1 = data1.detach().numpy()
        num = self.num
        num_rev = self.num_rev
        # Cov
        conv1 = self.ts_cov4d(data1, num, self.stride).to(torch.float)
        conv4 = self.ts_corr4d(data1, num, self.stride).to(torch.float)
        conv6 = self.ts_decaylinear(data1, self.stride).to(torch.float)
        conv7 = self.ts_std(data1, self.stride).to(torch.float)
        conv8 = self.ts_mean(data1, self.stride).to(torch.float)
        conv9 = self.ts_zcore(data1, self.stride).to(torch.float)
        conv10 = self.ts_mul(data1, num, num_rev, self.stride).to(torch.float)
        conv_skew = self.ts_skew(data1, self.stride).to(torch.float)
        # conv_kurt = self.ts_kurt(data1, self.stride).to(torch.float)
        conv_bn = self.ts_return(data1, self.stride).to(torch.float)
        # BN
        batch1 = self.bc1(conv1)
        batch4 = self.bc4(conv4)
        batch6 = self.bc6(conv6)
        batch7 = self.bc7(conv7)
        batch8 = self.bc8(conv8)
        batch9 = self.bc9(conv9)
        batch10 = self.bc10(conv10)

        # # 因子出现梯度爆炸
        # batch_skew = self.bc_skew(conv_skew)
        # # batch_kurt = self.bc_kurt(conv_kurt)
        # batch_rn = self.bc_rn(conv_bn)

        # # 合成因子保存到本地
        # batch_corrCT = batch4[:, :, 39, :]
        # batch_corrVT = batch4[:, :, 94, :]
        # if is_trian:
        #     np.save(f'E:\【Intern】\AlphaNet\zz800SingleFactors\Incenption2_E{index}.npy', batch_corrCT.cpu().detach().numpy())
        #     np.save(f'E:\【Intern】\AlphaNet\zz800SingleFactors\Incenption2_E{index}.npy', batch_corrVT.cpu().detach().numpy())
        # else:
        #     np.save(f'E:\【Intern】\AlphaNet\zz800SingleFactors\Incenption2_test.npy', batch_corrCT.cpu().detach().numpy())
        #     np.save(f'E:\【Intern】\AlphaNet\zz800SingleFactors\Incenption2_test.npy', batch_corrVT.cpu().detach().numpy())

        concat = [batch1, batch4, batch6, batch7, batch8, batch9, batch10]  # batch_skew, batch_rn
        concat_fea = torch.cat(concat, axis=2)
        gru_input = concat_fea.squeeze(1).transpose(1, 0).transpose(2, 0)  # N*1*9*30 -> 30*100*9
        output, hn = self.gru(gru_input)
        h = hn[-(1 + int(self.bidirectional)):]
        x = torch.cat(h.split(1), dim=-1).squeeze(0)
        output_inp2 = self.bcGRU(x.reshape([1, 1, x.shape[0], x.shape[1]])).reshape([-1, 30])
        return output_inp2


class Embedding(nn.Module):
    """对原始的输入时序做两层的GRU编码，再残差连接到最后
    """

    def __init__(self):
        super(Embedding, self).__init__()
        self.gru = nn.GRU(9, 30, 2)  # 两层的GRU
        self.bidirectional = False  # no Bi-GRU

    def forward(self, data):
        data = data.squeeze(1).transpose(1, 0).transpose(2, 0)  # N*1*9*30 -> 30*100*9
        output, hn = self.gru(data)
        h = hn[-(1 + int(self.bidirectional)):]
        x = torch.cat(h.split(1), dim=-1).squeeze(0)
        return x  # N*30,每个股票编码成长度为30的向量


class AlphaNet(nn.Module):
    """ Alpha Net Model
    """

    def __init__(self, fc1_num, fc2_num, num, num_rev, dropout_rate, stride1, stride2):
        super(AlphaNet, self).__init__()
        self.num = num
        self.num_rev = num_rev
        self.fc1_num = fc1_num
        self.fc2_num = fc2_num
        # parallel inception
        self.Inception_1 = Inception1(num, num_rev, stride1)
        self.Inception_2 = Inception2(num, num_rev, stride2)
        # GRU embedding
        self.Embedding = Embedding()
        # two fully connected layer
        self.fcOne = nn.Linear(fc1_num, 1)
        self.fc1 = nn.Linear(fc1_num, fc2_num)
        self.fc2 = nn.Linear(fc2_num, 1)
        # activation function
        self.relu = nn.ReLU()
        # drop out layer
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        # nn.init.xavier_uniform_(self.fc1.weight)  # 全连接层weights初始化，xavier的均匀分布初始化
        # nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.trunc_normal_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)  # bias初始化
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, data, index, is_trian):
        data_1 = self.Inception_1(data, index, is_trian)  # N*486
        data_2 = self.Inception_2(data, index, is_trian)  # N*486
        # data_3 = self.Embedding(data)  # N*30
        # pool_cat = torch.cat([data_1, data_2, data_3], axis=1)
        pool_cat = torch.cat([data_1, data_2], axis=1)  # N*1056

        # Myself
        x = self.fc1(pool_cat)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # 华泰一个
        # x = self.fcOne(pool_cat)

        x = x.to(torch.float)
        return x

    # def forward(self, data, epoch_num, is_trian:bool=True):
    #     data_1 = self.Inception_1(data)  # N*486
    #     data_2 = self.Inception_2(data)  # N*486
    #     # data_3 = self.Embedding(data)  # N*30
    #     # pool_cat = torch.cat([data_1, data_2, data_3], axis=1)
    #
    #     pool_cat = torch.cat([data_1, data_2], axis=1) # N*1056
    #     test_ = pool_cat.detach().numpy()
    #     if is_trian:
    #         np.save(f'./temp_factors/train_factor{epoch_num}.npy', pool_cat.detach().numpy())
    #     else:
    #         np.save(f'./temp_factors/test_factor.npy', pool_cat.detach().numpy())
    #
    #     x = self.fc1(pool_cat)
    #     x = self.relu(x)
    #     x = self.dropout(x)
    #     x = self.fc2(x)
    #     x = x.to(torch.float)
    #     return x


class Factor_data(Dataset):
    """ Factor Data
    默认输入的时候就已经是tensor
    """

    def __init__(self, train_x, train_y):
        self.len = len(train_x)
        self.x_data = train_x
        self.y_data = train_y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def load_data():
    X_train = np.load('./Data/X_train.npy')
    y_train = np.load('./Data/y_train.npy')
    X_test = np.load('./Data/X_test.npy')
    y_test = np.load('./Data/y_test.npy')
    return (X_train, y_train, X_test, y_test)


def generate(N):
    col = []
    col_rev = []
    for i in range(1, N):
        for j in range(0, i):
            col.append([i, j])
            col_rev.append([j, i])
    return (col, col_rev)


def torch_loader(testx, testy, batch_size):
    test_data = Factor_data(testx, testy)
    test_loader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             shuffle=False)  # 不打乱数据集
    return test_loader


def np2torch(X_train, y_train, X_test, y_test, num_fea: int = 15):
    trainx = torch.from_numpy(np.array(X_train)).reshape(len(X_train), 1, num_fea, 30)  # transform to tensor
    trainy = torch.from_numpy(np.array(y_train)).reshape(len(y_train), 1)  # label for regression
    testx = torch.from_numpy(np.array(X_test)).reshape(len(X_test), 1, num_fea, 30)
    testy = torch.from_numpy(np.array(y_test)).reshape(len(y_test), 1)
    return (trainx, trainy, testx, testy)


def alpha_set(alphanet, optim, bias_list, weight_list):
    """ AlphaNet Model Set
    Model set including:
        - weight:
        - optimizer: Adam
        - criterion: MSELoss
    """
    for name, p in alphanet.named_parameters():
        if 'bias' in name:
            bias_list += [p]
        else:
            weight_list += [p]

    # # SGD 优化器
    # optimizer = optim.SGD(
    #     [{'params': weight_list, 'weight_decay': 1e-5},
    #      {'params': bias_list, 'weight_decay': 0}],
    #     lr=1e-4,
    #     momentum=0.9,
    # )

    # # Adam 优化器
    optimizer = optim.Adam(
        [{'params': weight_list, 'weight_decay': 1e-5},
         {'params': bias_list, 'weight_decay': 0}],
        lr=1e-4,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss().to(device)
    return (bias_list, weight_list, optimizer, criterion)


def plot_loss(train_loss, test_loss):
    """plot train & test loss
    """
    plt.subplots(figsize=(10, 4))  # 指定画布大小
    x = list(np.arange(1, len(test_loss) + 1))
    y_train = train_loss
    y_test = test_loss
    plt.plot(x, y_train, label='Loss on Train Set')
    plt.plot(x, y_test, label='Loss on Test Set')
    plt.xlabel('Batch Num', weight='bold')
    plt.ylabel('Loss', weight='bold')
    plt.title('Train & Test Loss',  weight='bold')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.show(block=True)