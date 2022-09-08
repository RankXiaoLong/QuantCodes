import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import rankdata
from matplotlib import ticker


def np_load(li_train_nm: list, li_factor: list = [], path: str = 'temp_factors'):
    for i in li_train_nm:
        temp = np.load(rf'.\{path}\{i}')
        if len(temp.shape) == 1:
            temp = temp.reshape([-1, 1])
        li_factor.append(temp)
    return np.vstack(li_factor)


def split_train_test(li_Data: list, train_rating: float):
    np_X = np_load(li_Data, path='Data', li_factor=[])
    if len(np_X.shape) == 3:
        np_X = np_X.reshape((np_X.shape[0], -1, np_X.shape[1], np_X.shape[2]))
    else:
        np_X = np_X.reshape((np_X.shape[0], -1))
    cut_ = int(np.ceil(np_X.shape[0] * train_rating))
    X_train, X_test = np_X[: cut_], np_X[cut_:]
    return (X_train, X_test)


def get_X_y(li_Data_X: list, li_Data_Y: list, train_rating: float = 0.7):
    """获得 X & Y"""
    X_train, X_test = split_train_test(li_Data_X, train_rating)
    y_train, y_test = split_train_test(li_Data_Y, train_rating)
    return (X_train, X_test, y_train, y_test)


def load_fac(path_factor: str, path: str):
    """ 加载因子
    """
    li_factor_nm = os.listdir(path_factor)
    li_train_nm = [i for i in li_factor_nm if i.startswith('train')]
    li_train_nm.sort(key=lambda x: int("".join(filter(str.isdigit, x))))  # 对数字进行排序
    li_test_nm = list(set(li_factor_nm) - set(li_train_nm))
    li_train_fac = np_load(li_train_nm, path=path)
    li_test_fac = np_load(li_test_nm, li_factor=[], path=path)
    return (li_train_fac, li_test_fac)


def set_yaxis_formatter(ax, level: float = 1000.0, decimal_label: str = '1.1f', level_label: str = ' K'):
    """ 绘图标签设置"""
    # Format thousands e.g 10000 to 10.0K
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: format(x / level, decimal_label) + level_label))


def set_group(x: pd.Series, group: int = 10, method='first'):
    """获得组号, 用于groupby后的apply函数中
    1: 分组最小；
    group:分为几组"""
    size = x.size - np.count_nonzero(np.isnan(x))
    return np.ceil(x.rank(method=method) / (size / group))


def get_ec(stock: pd.DataFrame, fac: str, ret_next: str = 'NEXT_RET', group: int = 10):
    """ec: equity curve
    """
    stock['group'] = stock[fac].groupby(stock.date).apply(set_group, group=group)
    stock['group'] = stock['group'].astype(int)
    ec = stock.groupby(['date', 'group'])[ret_next].agg('mean').reset_index()
    return ec  # ec[ec.group == group][ret_next]


def get_stock_timeArray(path_Data: str):
    li_Data = os.listdir(path_Data)
    li_Data = [i for i in li_Data if i.startswith('20') if 'y' in i]
    dict_StockNUM = {}
    li_time = []
    for i in li_Data:
        temp = np.load(f'./Data/{i}')
        dict_StockNUM[i[:8]] = len(temp)

    for key in dict_StockNUM.keys():
        li_time.append(np.repeat(key, dict_StockNUM[key]).reshape(-1, 1))
    array_time = np.vstack(li_time)
    return array_time


def plot_ec(ec_: pd.DataFrame, ret_next: str = 'NEXT_RET', fac: str = 'TOTAL_PROFIT_GROWTH_RATE'):
    '''绘制单变量分组中各组的资金曲线
    ret_next: 分组数据中下期收益率的列名
    fac: 分组的变量的名称
    '''
    ec = ec_.copy()
    ec = ec.set_index('date').dropna()
    ec['equity_curve'] = ec.groupby('group')[ret_next].apply(lambda x: (x + 1).cumprod())
    temp_df = pd.pivot_table(ec, index='date', columns='group', values='equity_curve')
    lab = get_lab(fac=fac, group=int(ec.group.max()))
    temp_df.columns = lab.values()
    # temp_df.plot(figsize=(18, 8))
    group_ = print_info(temp_df)
    return temp_df, ec, group_


def print_info(temp_df: pd.DataFrame):
    """ 打印分组信息"""
    print('第一组最终收益：', temp_df.iloc[-1, 0])
    print('最后一组最终收益：', temp_df.iloc[-1, -1])
    temp_df = temp_df.stack().reset_index()
    group_ = temp_df.iloc[np.argmax(temp_df.iloc[:, 2]), 1]
    print('走势最强的组是: ', group_, '; 最终收益：', max(temp_df.iloc[:, 2]))
    return group_


def get_lab(fac: str, group: int = 10):
    """return dict including labels
    """
    lab = {}
    for i in range(group):
        if (i == 0):
            lab[i] = 'low-' + fac
        elif (i == group - 1):
            lab[i] = 'high-' + fac
        else:
            lab[i] = 'g' + str(i + 1)
    return lab


def neutralization(data: pd.DataFrame) -> pd.DataFrame:
    '''按市值、行业进行中性化处理 ps:处理后无行业市值信息'''
    factor_name = [i for i in data.columns.tolist() if
                   i not in ['INDUSTRY_CODE', 'market_cap', 'next_rn', 'date', 'codes']]

    # 回归取残差
    def _calc_resid(x: pd.DataFrame, y: pd.Series) -> float:
        result = sm.OLS(y, x).fit()
        return result.resid

    X = pd.get_dummies(data['INDUSTRY_CODE'])
    # 总市值单位为亿元
    X['market_cap'] = np.log(data['market_cap'] * 100000000)
    df = pd.concat([_calc_resid(X.fillna(0), data[i].astype(float)) for i in factor_name], axis=1)
    df.columns = factor_name
    df['INDUSTRY_CODE'] = data['INDUSTRY_CODE']
    df['market_cap'] = data['market_cap']
    df['next_rn'] = data['next_rn']
    return df


def evaluate(top_strategy: pd.DataFrame, next_rn: str = 'next_rn', Days: int = 10):
    """ 回测函数 """
    mean_ = np.exp(top_strategy[[next_rn]].mean() / Days * 252) - 1  # mean for daily
    std_ = (top_strategy[[next_rn]].apply(np.exp) - 1).std() / Days * 252 ** 0.5  # std for daily
    top_strategy['cumret'] = top_strategy[next_rn].cumsum().apply(np.exp)
    top_strategy['cummax'] = top_strategy['cumret'].cummax()
    top_strategy['drawdown'] = top_strategy['cummax'] - top_strategy['cumret']
    max_ = top_strategy['drawdown'].abs().max()
    win_ratio = top_strategy.query(f'{next_rn} > 0 ').shape[0] / top_strategy.shape[0]
    rf = 0.03
    sharpe = (mean_ - rf) / std_
    print('年化收益率', mean_.values)
    print('年化波动率', std_.values)
    print('胜率', win_ratio)
    print('最大回撤', max_)
    print('Sharpe value', sharpe.values)

    analyze = pd.DataFrame()  # 用于存储计算的指标
    return_df = top_strategy[next_rn]
    # -------------------------------------------------------
    # part2:计算年化收益率
    # -------------------------------------------------------
    cumulative = np.exp(np.log1p(return_df).cumsum())  # 计算整个回测期内的复利收益率
    cum_return = cumulative - 1
    annual_return_df = (1 + cum_return) ** (252 / len(return_df) / Days) - 1  # 计算年化收益率
    analyze['annual_return'] = annual_return_df.iloc[-1]  # 将年化收益率的Series赋值给数据框

    cumulative = np.exp(np.log1p(return_df).cumsum()) * 100  # 计算整个回测期内的复利收益率
    max_return = cumulative.cummax()  # 计算累计收益率的在各个时间段的最大值
    analyze['max_drawdown'] = cumulative.sub(max_return).div(max_return).min()  # 最大回撤一般小于0，越小，说明离1越远，各时间点与最大收益的差距越大


def get_performance(hs300DiffStrategy: pd.DataFrame, columns: list, benchmark_col: str = 'HS300', Days: int = 10):
    """ 计算单因子回测指标
    程序中使用的回测函数--大概包含8步
    """
    # -------------------------------------------------------
    # 准备数据
    # -------------------------------------------------------
    return_df = hs300DiffStrategy[columns]
    analyze = pd.DataFrame()  # 用于存储计算的指标
    # -------------------------------------------------------
    # part2: 计算年化收益率
    # -------------------------------------------------------
    cumulative = np.exp(np.log1p(return_df).cumsum())  # 计算整个回测期内的复利收益率
    cum_return = cumulative - 1
    annual_return_df = (1 + cum_return) ** (252 / len(return_df) / Days) - 1  # 计算年化收益率
    analyze['annual_return'] = annual_return_df.iloc[-1]  # 将年化收益率的Series赋值给数据框
    # -------------------------------------------------------
    # part3:计算收益波动率（以年为基准）
    # return中的收益率为日收益率，所以计算波动率转化为年时，需要乘上np.sqrt(252)
    # -------------------------------------------------------
    (hs300DiffStrategy[columns].apply(np.exp) - 1).std() / Days * 252 ** 0.5  # std for daily
    analyze['return_volatility'] = return_df.std() * np.sqrt(252 / Days)

    # -------------------------------------------------------
    # part4:计算夏普比率
    # -------------------------------------------------------
    risk_free = 0.00
    # return_risk_adj = return_df - risk_free
    # analyze['sharpe_ratio'] = return_risk_adj.mean() / np.std(return_risk_adj, ddof=1)
    analyze['sharpe_ratio'] = (analyze['annual_return'] - risk_free) / analyze['return_volatility']
    # -------------------------------------------------------
    # part5: 最大回撤
    # -------------------------------------------------------
    cumulative = cumulative * 100
    max_return = cumulative.cummax()  # 计算累计收益率的在各个时间段的最大值
    drawdown = cumulative.sub(max_return).div(max_return)
    analyze['max_drawdown'] = drawdown.min()  # 最大回撤一般小于0，越小，说明离1越远，各时间点与最大收益的差距越大

    # -------------------------------------------------------
    # part6:计算相对指标
    # -------------------------------------------------------
    analyze['relative_return'] = np.array(analyze.loc[columns, 'annual_return']) - np.array(
        analyze.loc[benchmark_col, 'annual_return'])  # 计算相对年化波动率
    analyze['relative_volatility'] = np.array(analyze.loc[columns, 'return_volatility']) - np.array(
        analyze.loc[benchmark_col, 'return_volatility'])  # 计算相对波动
    analyze['relative_drawdown'] = np.array(analyze.loc[columns, 'max_drawdown']) - np.array(
        analyze.loc[benchmark_col, 'max_drawdown'])  # 计算相对最大回撤

    # -------------------------------------------------------
    # part7:计算信息比率
    # 计算策略与基准日收益差值的年化标准差
    # -------------------------------------------------------
    # return_diff = return_df.sub(return_df[benchmark_col], axis=0).std() * np.sqrt(252 / Days)
    return_diff = (return_df[columns[0]] - return_df[benchmark_col]).std() * np.sqrt(252 / Days)
    analyze['info_ratio'] = analyze['relative_return'].div(return_diff)
    # -------------------------------------------------------
    # part8: 胜率
    # -------------------------------------------------------
    win_ratio = []
    NM = 'relative_rn'
    temp = (return_df.iloc[:, 0] - return_df.loc[:, benchmark_col]).to_frame().rename(columns={0: NM})
    win_ratio.append(temp.query(f"{NM} > 0").shape[0] / return_df.shape[0])
    win_ratio.append(return_df.query(f'{benchmark_col} > 0 ').shape[0] / return_df.shape[0])
    analyze['win_ratio'] = win_ratio
    return (analyze.T, drawdown)


def gey_Yperformance(hs300DiffStrategy: pd.DataFrame, columns: list = ['Strategy_next_rn', 'HS300'],
                     benchmark_col='HS300',
                     Days: int = 10):
    result_dic = {}  # 用于存储每年计算的各项指标
    if type(hs300DiffStrategy.index[0]) == str:
        hs300DiffStrategy.index = pd.to_datetime(hs300DiffStrategy.index)
    for y, df in hs300DiffStrategy.groupby(pd.Grouper(level='date', freq='Y')):
        result, _ = get_performance(df, columns=columns, benchmark_col=benchmark_col, Days=Days)
        result_dic[y.strftime('%Y')] = result.iloc[:, 0]
    result_df = pd.DataFrame(result_dic)
    return result_df.T


def rankWeightSeed(path_factor: str, path_factorNM: list, RankWeight: bool = True):
    """ rank 随机种子结果，归一化加权
    RankWeight: Ture, rank 加权；RankWeight: False, mean加权
    """
    for i, fac_name in enumerate(path_factorNM):
        if i == 0:
            fac = np.load(fr'{path_factor}\{fac_name}', allow_pickle=True)[:, 3].astype(np.float32).reshape(-1, 1)
        else:
            temp = np.load(fr'{path_factor}\{fac_name}', allow_pickle=True)[:, 3].astype(np.float32).reshape(-1, 1)
            fac = np.hstack([fac, temp])
    if RankWeight:
        rankfac = rankdata(fac, axis=1)
        fac = (fac * rankfac / rankfac[0, :].sum()).sum(axis=1).reshape(-1, 1)
    else:
        fac = fac.mean(axis=1).reshape(-1, 1)
    return fac, fac_name


def rankWeightIC(df_IC: pd.DataFrame, df_IC_index: pd.DataFrame.index, df_ec_Combine: pd.DataFrame,
                 columns=range(0, 60)):
    """ 对多个单因子进行rank排序，归一化后计算合成因子
    df_IC: 每个时期，对于每个因子存在一个IC值
    df_ec_Combine 中包含 columns,index 为 df_IC_index
    """
    # shift 1, only for method2, 计算与下一期的IC，根据ICrank加权，不能引入未来信息，需要shift(1)
    df_IC = df_IC.shift(1).dropna()
    weight_rankData = rankdata(df_IC, axis=1)
    weight_rankData = weight_rankData / np.sum(weight_rankData[0, :])
    rankICfac = []  # 存储rank权重合成因子
    for i, date in enumerate(df_IC_index):
        temp_ec = df_ec_Combine.query('index == @date')
        # 去市值、行业, range(0, 60) 是因子对应的列名
        # n*60 与 60*1 进行广播运算，
        temp_ec = (temp_ec.loc[:, columns] * weight_rankData[i, :]).sum(axis=1).values.reshape(-1, 1)  # rankIC 加权
        rankICfac.append(temp_ec)
    array_rankICfac = np.vstack(rankICfac)
    # combine_df = pd.DataFrame(array_rankICfac, columns=['comb_fac'], index=df_ec_Combine.index)
    return pd.DataFrame(array_rankICfac, columns=['comb_fac'], index=df_ec_Combine.index)


# plot
def plot_performance(plot_df: pd.DataFrame, fig_title: str, save_path: str):
    """ 绘制净值曲线"""
    ax = plot_df.plot(figsize=(16, 8))
    plt.title(fig_title, fontsize='14', weight='bold')
    plt.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5)
    plt.xlabel(' ')
    set_yaxis_formatter(ax, level=0.01, decimal_label='1.0f', level_label=' %')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=500)
    plt.show(block=True)  # 防止 matplotlib长期未响应

def align_index(df_hs300:pd.DataFrame, df_ec:pd.DataFrame, LastDate:str):
    """" 找出与策略的换仓日期对应的大盘指数的下一期的收益率
    df_hs300.head()
         ts_code trade_date     close
    0  399300.SZ   20100104  3535.229
    1  399300.SZ   20100105  3564.038
    2  399300.SZ   20100106  3541.727
    3  399300.SZ   20100107  3471.456
    4  399300.SZ   20100108  3480.130
    """
    df_hs300 = df_hs300.sort_values(by=['trade_date']).reset_index(drop=True)
    Date_stocks = df_ec.index.unique().tolist()  # 日期对在一起
    Date_stocks.append(LastDate)
    df_hs300_bet10days = df_hs300.query('trade_date in (@Date_stocks)').reset_index(drop=True)
    df_hs300_bet10days['rn'] = df_hs300_bet10days.close.pct_change().shift(-1)
    return df_hs300_bet10days.dropna()

def diffIndex(top_strategy:pd.DataFrame, df_hs300_bet10days:pd.DataFrame):
    """ 策略与大盘基准指数比较
    hs300DiffStrategy.columns 包括 ['Strategy_next_rn', 'HS300'], values 为下一期收益率
    plot_diff_rn.columns 包括 ['Strategy_next_rn', 'HS300'], values 为累计净值
    """
    hs300DiffStrategy = pd.DataFrame(index=top_strategy.date)
    hs300DiffStrategy['Strategy'] = top_strategy.next_rn.values
    hs300DiffStrategy['HS300'] = df_hs300_bet10days.next_rn.values

    plot_diff_rn = pd.DataFrame(index=top_strategy.date)
    plot_diff_rn['Strategy'] = ((1 + top_strategy.next_rn).cumprod() - 1).values
    plot_diff_rn['HS300'] = ((1 + df_hs300_bet10days.next_rn).cumprod() - 1).values
    return (hs300DiffStrategy, plot_diff_rn)


def transform_group(group_: str, group: int):
    """ 将group_(文字) 映射到对应的数字组
    group为对应的最大分组数"""
    if group_ == 'high-comb_fac':
        return group
    elif group_ == 'low-comb_fac':
        return 1
    else:
        return int("".join(filter(str.isdigit, group_)))
