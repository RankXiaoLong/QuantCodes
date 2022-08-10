#
# Ref: https://zhuanlan.zhihu.com/p/546110583
# 主要来自华泰研报: <因子挖掘和神经网络>
#

# import datetime
# import math
# import time
# from datetime import timedelta
# import pandas as pd
# import statsmodels.api as sm
# import matplotlib
# from torch.utils.data import Dataset
# import torch.nn.functional as F
# import tushare as ts
# import tools as tl
# import numpy as np
# import torch.nn as nn

from matplotlib import pyplot as plt
from tqdm import tqdm
import torch

from torch import optim
import model as ml  # local

if __name__ == '__main__':
    (X_train, y_train, X_test, y_test) = ml.load_data()

    print("Training samples: ", X_train.shape[0])
    print("Testing samples: ", X_test.shape[0])
    print("\n")

    batch_size = 1000
    (trainx, trainy, testx, testy) = ml.np2torch(X_train, y_train, X_test, y_test)

    # put into data loader
    train_loader = ml.torch_loader(trainx, trainy, batch_size)
    test_loader = ml.torch_loader(trainx, trainy, batch_size)

    # Building the model
    num, num_rev = ml.generate(9)
    alphanet = ml.AlphaNet(1002, 30, num, num_rev, 0.5)

    # weight decay: 对所有weight参数进行L2正则化
    weight_list, bias_list = [], []
    (bias_list, weight_list, optimizer, criterion) = ml.alpha_set(alphanet, optim, bias_list, weight_list)

    # # training
    epoch_num = 5
    loss_list = []
    test_loss = []
    for epoch in range(epoch_num):
        for data, label in tqdm(train_loader, f'AlphaNet-epoch {epoch}'):
            # 训练模式
            alphanet.train()  # training pattern, grad required.
            out_put = alphanet(data.to(torch.float))
            loss = criterion(out_put, label.to(torch.float))
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # renew the parameters

            # 测试模式
            alphanet.eval()
            y_pred = alphanet(testx.to(torch.float))
            testloss = criterion(y_pred, testy)
            test_loss.append(testloss.item())

        # print("current epoch time:",epoch+1)
        # print("current loss of epoch ",epoch+1,":", train_loss.item())

    # Visualization
    with plt.style.context(['seaborn']):
        ml.plot_loss(test_loss)
