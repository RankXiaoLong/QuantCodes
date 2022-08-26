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

import model
import model as ml  # local
import combine_ic as ci
import os
import datetime


if __name__ == '__main__':

    # ----------------------------------------------------
    # path
    # ----------------------------------------------------
    path_factor = r'E:\【Intern】\AlphaNet\temp_factors'
    path_stock = r'E:\【Intern】\AlphaNet\stock_lists'
    path_Data = r'E:\【Intern】\AlphaNet\Data'

    # li_factor_nm = os.listdir(path_factor)
    # li_stock = os.listdir(path_stock)
    # li_Data = os.listdir(path_Data)
    # li_Data_X = [i for i in li_Data if i.startswith('20') if '_X_' in i]
    # li_Data_Y = [i for i in li_Data if i.startswith('20') if '_y_' in i]

    # ----------------------------------------------------
    # get train/test
    # ----------------------------------------------------
    (X_train, y_train, X_test, y_test) = ml.load_data()
    # (X_train, X_test, y_train, y_test) = ci.get_X_y(li_Data_X, li_Data_Y, train_rating=0.7)

    print("Training samples: ", X_train.shape[0])
    print("Testing samples: ", X_test.shape[0])
    print("\n")

    batch_size = 1000
    (trainx, trainy, testx, testy) = ml.np2torch(X_train, y_train, X_test, y_test)

    # put into data loader
    train_loader = ml.torch_loader(trainx, trainy, batch_size)
    test_loader = ml.torch_loader(testx, testy, batch_size)

    # Building the model
    num, num_rev = ml.generate(9)
    alphanet = ml.AlphaNet(894, 30, num, num_rev, 0.5, stride1=10, stride2=3)

    # weight decay: 对所有weight参数进行L2正则化
    weight_list, bias_list = [], []
    (bias_list, weight_list, optimizer, criterion) = ml.alpha_set(alphanet, optim, bias_list, weight_list)

    # # training
    epoch_num = 2
    train_loss_li = []
    test_loss_li = []

    for epoch in range(epoch_num):
        i = 1
        train_loss, test_loss = 0, 0
        for data, label in tqdm(train_loader, f'AlphaNet-epoch {epoch}'):
            # 训练模式
            alphanet.train()  # training pattern, grad required.
            out_put = alphanet(data.to(torch.float), i, is_trian=True)
            loss = criterion(out_put, label.to(torch.float))
            train_loss += loss.item()
            # loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # renew the parameters

            # 测试模式
            alphanet.eval()
            y_pred = alphanet(testx.to(torch.float), i, is_trian=False)
            test_loss = criterion(y_pred, testy)
            # test_loss.append(testloss.item())
            test_loss += test_loss.item()
            i += 1

        train_loss_li.append(train_loss)
        test_loss_li.append(test_loss.item())
        # print("current epoch time:",epoch+1)
        # print("current loss of epoch ",epoch+1,":", train_loss.item())
    today = datetime.date.today()
    torch.save(alphanet, f'./save_models/alphanet_model_{today}.pth')
    # model = torch.load(f'./save_models/alphanet_model_{today}.pth')
    # model.eval()
    # y_pred_test = model(testx.to(torch.float), i, is_trian=False)

    # Visualization
    with plt.style.context(['seaborn']):
        ml.plot_loss(train_loss_li, test_loss_li)
