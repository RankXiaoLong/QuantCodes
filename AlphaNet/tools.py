# import datetime
# import math
# import time
# import statsmodels.api as sm
# import matplotlib
# from matplotlib import pyplot as plt
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from torch import optim
# import torch.nn.functional as F

from datetime import timedelta
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import tushare as ts
import time
import os


token = '4df6fb27a32955fd6af41474d045d906468e94d3a3c7838244b46ca5'
ts.set_token(token)
pro = ts.pro_api()

# 给定一个交易日，返回该日满足条件的A股股票列表
def get_stocklist(date: str, Num: int = 1000):
    start = (pd.to_datetime(date) - timedelta(30)).strftime('%Y%m%d')
    df1 = pro.index_weight(index_code='000002.SH', start_date=start, end_date=date)  # 交易日当天的股票列表
    codes = df1['con_code'].to_list()[0:Num] # 在每个截面期只选取1200只股票
    return codes


# 给定日期区间的端点，输出期间的定长采样交易日列表
# 默认定长采样是10，即每隔10天获取一次数据，包括过去30天的数据，和未来10天的收益率
def get_datelist(start: str, end: str, Interval: int = 10):
    df = pro.index_daily(ts_code='399300.SZ', start_date=start, end_date=end)
    date_list = list(df.iloc[::-1]['trade_date'])
    sample_list = [date_list[i] for i in range(len(date_list)) if i % Interval == 0]
    return sample_list

def PastFutureDays(Date, Pass_day, Future_day):
    start = (pd.to_datetime(Date) - timedelta(Pass_day * 2)).strftime('%Y%m%d')
    end = (pd.to_datetime(Date) + timedelta(Future_day * 2)).strftime('%Y%m%d')
    return (start, end)


# 返回两个，一个是前30个交易日的9个指标面板（9*30），一个是未来10天的收益率
def get_x_y(code: str, Date: str, Pass_day: int, Future_day: int, len1: int, len2: int):
    (start, end) = PastFutureDays(Date, Pass_day, Future_day)
    df_price = pro.daily(ts_code=code,  # OHLC,pct_change,volume
                         start_date=start, end_date=Date)
    df_basic = pro.daily_basic(ts_code=code, start_date=start, end_date=Date)
    df_return = pro.daily(ts_code=code, start_date=Date, end_date=end).iloc[::-1]['close']

    if (df_price.shape[0] == df_basic.shape[0]) & (df_price.shape[0] == len1) & (
            df_return.shape[0] == len2):  # 判断数据的完整性
        df_price = df_price.iloc[0:Pass_day, [2, 3, 4, 5, 8, 9]].fillna(0.1)
        df_basic = df_basic.iloc[0:Pass_day, [3, 4, 5]].fillna(0.1)
        data = np.array(pd.merge(df_price, df_basic, left_index=True, right_index=True).iloc[::-1].T)
        # 未来十个交易日的收益率
        dfr = df_return.iloc[0:Future_day]
        ret = dfr.iloc[-1] / dfr.iloc[0] - 1  # 后十个交易日的收益率
        return data, ret
    else:
        return None, None  # 数据缺失的预处理

def get_length(Date: str, Pass_day: int, Future_day: int):
    """获得长度：
    第一个为；过去2个Pass_day 中交易日的天数；
    第二个为：未来2个Future_day 中交易的天数 .

    为什么设置2倍: 因为存在周末、非交易日等
    """
    (start, end) = PastFutureDays(Date, Pass_day, Future_day)
    len_1 = pro.index_daily(ts_code='399300.SZ', start_date=start, end_date=Date).shape[0]
    len_2 = pro.index_daily(ts_code='399300.SZ', start_date=Date, end_date=end).shape[0]
    return len_1, len_2


# 构造数据集的函数：输入一个时间区间的端点，得到该区间内采样交易日期的所有数据
# 默认使用过去30天的数据预测未来10天的数据，过去30天的数据每隔10天采样一次
def get_dataset(start: str, end: str, Interval: int = 10, Pass_day: int = 30, Future_day: int = 10):
    trade_date_list = get_datelist(start, end, Interval)
    for i_, date in enumerate(trade_date_list):
        stock_list = get_stocklist(date)
        len1, len2 = get_length(date, Pass_day, Future_day)

        index = range(len(stock_list))
        X_train = []
        y_train = []

        for i in tqdm(index, f'get_dataset {date}'):

            code = stock_list[i]
            x, y = get_x_y(code, date, Pass_day, Future_day, len1, len2)
            try:
                if (x.shape[0] == 9) & (x.shape[1] == Pass_day):
                    X_train.append(x)
                    y_train.append(y)
            except Exception:
                continue
        if len(y_train) > 0:
            np.save(f'./Data/{date}_X_train.npy', np.array(X_train))
            np.save(f'./Data/{date}_y_train.npy', np.array(y_train))
    # return X_train, y_train

def get_codes(start: str, end: str, Interval: int = 10, Pass_day: int = 30, Future_day: int = 10):
    trade_date_list = get_datelist(start, end, Interval)
    for i_, date in tqdm(enumerate(trade_date_list), desc='get codes'):
        print(f'{date}', end=' ')
        stock_list = get_stocklist(date)
        if len(stock_list) > 0:
            np.save(f'./stock_lists/{date}_stock_lists.npy', np.array(stock_list))


def get_delStock(start: str, end: str, Interval: int = 10, Pass_day: int = 30, Future_day: int = 10):
    trade_date_list = get_datelist(start, end, Interval)
    for i_, date in enumerate(trade_date_list):
        stock_list = get_stocklist(date)
        len1, len2 = get_length(date, Pass_day, Future_day)

        index = range(len(stock_list))
        stock_del = []

        for i in tqdm(index, f'get_del_stock {date}'):

            code = stock_list[i]
            x = get_stock_x(code, date, Pass_day, Future_day, len1, len2)
            try:
                if (x.shape[0] == 9) & (x.shape[1] == Pass_day):
                    continue
            except Exception:
                stock_del.append(code)
                continue
        if len(stock_del) > 0:
            np.save(f'./stock_del/{date}_stock_del.npy', np.array(stock_del))

def print_len_name(path_data: str, path_codes: str):
    data_name = [i for i in os.listdir(path_data) if i.startswith('20')]
    code_names = [i for i in os.listdir(path_codes) if i.startswith('20')]

    for i in range(len(code_names)):
        X_train = np.load(f'./Data/{data_name[i]}')
        code_name = np.load(f'./stock_lists/{code_names[i]}')
        print(code_names[i], len(X_train))
        print(code_names[i], len(code_name))


# 返回两个，一个是前30个交易日的9个指标面板（9*30），一个是未来10天的收益率
def get_stock_x(code: str, Date: str, Pass_day: int, Future_day: int, len1: int, len2: int):
    (start, end) = PastFutureDays(Date, Pass_day, Future_day)
    df_price = pro.daily(ts_code=code,  # OHLC,pct_change,volume
                         start_date=start, end_date=Date)
    df_basic = pro.daily_basic(ts_code=code, start_date=start, end_date=Date)
    df_return = pro.daily(ts_code=code, start_date=Date, end_date=end).iloc[::-1]['close']

    if (df_price.shape[0] == df_basic.shape[0]) & (df_price.shape[0] == len1) & (
            df_return.shape[0] == len2):  # 判断数据的完整性
        df_price = df_price.iloc[0:Pass_day, [2, 3, 4, 5, 8, 9]].fillna(0.1)
        df_basic = df_basic.iloc[0:Pass_day, [3, 4, 5]].fillna(0.1)
        data = np.array(pd.concat([df_price, df_basic], axis=1).iloc[::-1].T)
        return data
    else:
        return None  # 数据缺失的预处理

def align_wind(df_stock_basic):
    """ JQ 数据对齐wind格式 """
    code_tushare = []
    for code in df_stock_basic['codes'].values:
        if '.XSHE' in code:
            code_tushare.append(code.replace('.XSHE', '.SZ'))
        else:
            code_tushare.append(code.replace('.XSHG', '.SH'))
    df_stock_basic['codes'] = code_tushare
    df_stock_basic['date'] = df_stock_basic['date'].apply(lambda x: "".join(filter(str.isdigit, x)))
    return df_stock_basic

def dict2df(stock_dic_1600):
    """ dict 转 pd.DataFrame"""
    list_stoc = []
    list_date = []
    for d in tqdm(stock_dic_1600.keys()):
        list_stoc.append(np.array(stock_dic_1600[d]).reshape(-1, 1))
        list_date.append(np.repeat(d, len(stock_dic_1600[d])).reshape(-1, 1))

    stock_array = np.vstack(list_stoc)
    date_array = np.vstack(list_date)
    df = pd.DataFrame(np.hstack([date_array, stock_array]), columns=['date', 'codes'])
    return df