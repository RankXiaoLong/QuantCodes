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
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
from torch import optim
import model as ml  # local
import combine_ic as ci
import os
import datetime
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def rollingIndex(df, Q_date, len_cut:int=5, STRAT_DATE:str ='20100101'):
    """
    STRAT_DATE = '20100101'
    len_cut=5: # 4:1 trian:4个季度，test: 1个季度
    """
    # 初始参数设置，保留对应的index；
    STRAT_DATE = STRAT_DATE
    TrainIndex = []
    TestIndex = []
    date_cut_ = [i * len_cut for i in range(1, int(np.floor(len(Q_date) / len_cut)) + 1)]
    range_time = [(Q_date[cut_ - 2], Q_date[cut_ - 1]) for cut_ in date_cut_]

    # for loop
    for i, date in enumerate(range_time):
        train_EndDate = date[0]
        test_EndDate = date[1]
        if i == 0:
            train_index = df.query('@STRAT_DATE <= date <= @train_EndDate').index.values.reshape(-1, 1)
            test_index = df.query('@train_EndDate < date <= @test_EndDate').index.values.reshape(-1, 1)
            TrainIndex.append(train_index)
            TestIndex.append(test_index)
        else:
            train_index = df.query('@last_testdate < date <= @train_EndDate').index.values.reshape(-1, 1)
            test_index = df.query('@train_EndDate < date <= @test_EndDate').index.values.reshape(-1, 1)
            TrainIndex.append(train_index)
            TestIndex.append(test_index)
        last_testdate = test_EndDate
    if len(Q_date) %  len_cut != 0:
        train_EndDate = Q_date[-2]
        test_EndDate = Q_date[-1]
        train_index = df.query('@last_testdate <= date <= @train_EndDate').index.values.reshape(-1, 1)
        test_index = df.query('@train_EndDate <= date <= @test_EndDate').index.values.reshape(-1, 1)
        TrainIndex.append(train_index)
        TestIndex.append(test_index)
    return (np.vstack(TrainIndex), np.vstack(TestIndex))


if __name__ == '__main__':

    # ----------------------------------------------------
    # path
    # ----------------------------------------------------
    # path_factor = r'E:\【Intern】\AlphaNet\temp_factors'
    # path_stock = r'E:\【Intern】\AlphaNet\stock_lists'
    # path_Data = r'E:\【Intern】\AlphaNet\Data'
    # path_Wind = r'E:\【Intern】\AlphaNet\wind_data'
    # li_npy = os.listdir(path_Wind)

    # seed: 2020, 2022, 567, 123, 2000, 1900, 1000, 800, 666, 1999
    seeds = [2020, 2022, 567, 123, 2000, 1900, 1000, 800, 666, 1999]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\n')
    print('*' * 30)
    print('device', device)
    print('*' * 30)
    for seed in seeds[:]:
        print('seeds', seed)
        print('*' * 30)
        # seed = seeds[0]
        setup_seed(seed)
        # -------------------------------------------------------
        # rolling train & test or CUT by time
        # --------------------------------------------------------
        # Method 1: rolling
        # ZZ800: 'E:\【Intern】\AlphaNet\OracleData\ZZ800Wind\Merge Data'
        # AMarkets: 'E:\【Intern】\AlphaNet\OracleData\AMarket\Merge Data'
        path = r'E:\【Intern】\AlphaNet\OracleData\AMarket\Merge Data'
        array_X15 = np.load(f'{path}\X_15.npy', allow_pickle=True).astype(np.float32)
        array_Y1 = np.load(f'{path}\Y_1.npy', allow_pickle=True).astype(np.float32)
        array_Y3 = np.load(fr'{path}\rn_TS_Y_3.npy', allow_pickle=True)
        # array_Y1 = array_Y1 * 100

        date = pd.to_datetime(np.unique(array_Y3[:, 0])).to_frame().rename(columns={0: 'date'})
        # 转化为季度日期: Q; 年度日期: Y;
        Q_date = [i.strftime('%Y%m%d') for i in date.resample('Y', on="date").last().index]
        df = pd.DataFrame(array_Y3, columns=['date', 'code', 'rn'])
        df.iloc[:, 2] = df.iloc[:, 2].astype(np.float32)
        TrainIndex, TestIndex = rollingIndex(df, Q_date, len_cut=4)


        X_train = array_X15[TrainIndex].reshape(-1, 1, 15, 30)
        y_train = array_Y1[TrainIndex].reshape(-1, 1)
        X_test = array_X15[TestIndex].reshape(-1, 1, 15, 30)
        y_test = array_Y1[TestIndex].reshape(-1, 1)

        # # Method 2: cut by time
        # CUT = int(np.ceil(array_Y1.shape[0] * 0.7))
        # X_train, X_test = array_X15[: CUT], array_X15[CUT:]
        # y_train, y_test = array_Y1[: CUT], array_Y1[CUT:]

        # ----------------------------------------------------
        # get train/test
        # ----------------------------------------------------
        # (X_train, y_train, X_test, y_test) = ml.load_data()
        # (X_train, X_test, y_train, y_test) = ci.get_X_y(li_Data_X, li_Data_Y, train_rating=0.7)

        print("Training samples: ", X_train.shape[0])
        print("Testing samples: ", X_test.shape[0])
        print('*' * 30)
        print("\n")

        batch_size = 6000
        (trainx, trainy, testx, testy) = ml.np2torch(X_train, y_train, X_test, y_test)

        # put into data loader
        train_loader = ml.torch_loader(trainx, trainy, batch_size)
        test_loader = ml.torch_loader(testx, testy, batch_size)

        # Building the model
        num, num_rev = ml.generate(15)
        alphanet = ml.AlphaNet(60, 30, num, num_rev, 0.5, stride1=10, stride2=5)
        alphanet = alphanet.to(device)  # 利用GPU

        # weight decay: 对所有weight参数进行L2正则化
        weight_list, bias_list = [], []
        (bias_list, weight_list, optimizer, criterion) = ml.alpha_set(alphanet, optim, bias_list, weight_list)

        # # training
        epoch_num = 100
        train_loss_li = []
        test_loss_li = []
        best_test_epoch, best_test_loss = 0, np.inf
        today = datetime.date.today()
        for epoch in range(epoch_num):
            i = 1
            train_loss, test_loss = 0, 0
            # 训练模式
            alphanet.train()  # training pattern, grad required.
            for data, label in tqdm(train_loader, f'AlphaNet-epoch {epoch}/train '):
                data, label = data.to(device, dtype=torch.float), label.to(device, dtype=torch.float)
                out_put = alphanet(data, i, is_trian=True)
                loss = criterion(out_put, label)
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  # renew the parameters

            # 测试模式
            alphanet.eval()
            with torch.no_grad():
                for data, label in tqdm(test_loader, f'AlphaNet-epoch {epoch}/test'):
                    data, label = data.to(device, dtype=torch.float), label.to(device, dtype=torch.float)
                    y_pred = alphanet(data, i, is_trian=False)
                    test_loss = criterion(y_pred, label)
                    test_loss += test_loss.item()

            i += 1
            train_loss_li.append(train_loss)
            test_loss_li.append(test_loss.item())

            if test_loss < best_test_loss:
                torch.save(alphanet, f'./BestModels/Best_{seed}_alphanet_model_{today}.pth')
                best_test_loss = test_loss
                best_test_epoch = epoch
            if epoch - best_test_epoch > 10:
                print(f'{seed}_best_test_loss', best_test_loss)
                print(f'{seed}_best_test_epoch', best_test_epoch)
                break
            # print("current epoch time:",epoch+1)
            # print("current loss of epoch ",epoch+1,":", train_loss.item())

        #   -------------------------------------------------------------------
        #   save model
        #   -------------------------------------------------------------------
        # torch.save(alphanet, f'./save_models/{seed}_alphanet_model_{today}.pth')
        # -------------------------------------------------------------------
        #  load models
        # -------------------------------------------------------------------
        # today = '2022-08-27'
        # model = torch.load(f'./save_models/2020_alphanet_model_2022-08-30.pth')
        # y_train = model(trainx.to(torch.float), i, is_trian=True)

        y_pred_train = []
        y_pred_test = []
        i = 0
        alphanet.eval()
        with torch.no_grad():
            for i, (data, label) in tqdm(enumerate(train_loader), f'AlphaNet-epoch pre/train '):
                data, label = data.to(device, dtype=torch.float), label.to(device, dtype=torch.float)
                train_out_put = alphanet(data, i+1, is_trian=True)
                y_pred_train.append(train_out_put)

            for  i, (data, label) in tqdm(enumerate(test_loader), f'AlphaNet-epoch pre/test'):
                data, label = data.to(device, dtype=torch.float), label.to(device, dtype=torch.float)
                test_out_put = alphanet(data, i+1, is_trian=False)
                y_pred_test.append(test_out_put)
        y_pred_train = np.array(torch.vstack(y_pred_train).cpu())
        y_pred_test = np.array(torch.vstack(y_pred_test).cpu())

        # 按照method1的拼在一起
        train_pre = np.concatenate([TrainIndex, y_pred_train], axis=1)
        test_pre = np.concatenate([TestIndex, y_pred_test], axis=1)

        df_data_pred = pd.DataFrame(np.vstack([train_pre, test_pre]), columns=['index', 'pred'])
        df_data_pred['index'] = df_data_pred['index'].astype(int)
        df_array_Y3 = pd.DataFrame(array_Y3, columns=['date', 'codes', 'next_rn'])
        df_array_Y3 = df_array_Y3.reset_index()
        SyntaxFac = pd.merge(df_array_Y3, df_data_pred, how='left', on='index').drop(columns=['index'])
        np.save(fr'E:\【Intern】\AlphaNet\NewestDateSeeds\{seed}_SyntaxFactor.npy', SyntaxFac.values)
        # with plt.style.context(['seaborn']):
        #     ml.plot_loss(train_loss_li, test_loss_li)

        print('*' * 30)
        print(f'{seed}_best_test_loss', best_test_loss.item())
        print(f'{seed}_best_test_epoch', best_test_epoch)
        print('--finish--')
        print('*'*30)
        print('\n')