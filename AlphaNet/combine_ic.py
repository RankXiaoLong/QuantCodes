import numpy as np
import os
import pandas as pd
# import model as ml
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import rankdata
from matplotlib import ticker
import backtools as bl
pd.set_option('expand_frame_repr', False)
import datetime


if __name__ == '__main__':
    # os获取因子文件名
    path_factor = r'E:\【Intern】\AlphaNet\NewestDateSeeds'
    path_factorNM = os.listdir(path_factor)
    path_factorNM.sort(key=lambda x: int("".join(filter(str.isdigit, x))))
    # --------------------------------------------------------------------
    # 等权随机种子的合成因子，仅适用于method1: 输入的是合成因子，加权降低随机性的影响
    # --------------------------------------------------------------------
    # df_basic.columns: 'date', 'codes', 'ComFac', 与基准对齐（'date', 'codes'）
    fac, fac_name = bl.rankWeightSeed(path_factor, path_factorNM, RankWeight=True)
    df_basic = pd.DataFrame(np.load(fr'{path_factor}\{fac_name}', allow_pickle=True)[:, :2], columns=['date', 'codes'])
    df_basic['ComFac'] = fac

    # --------------------------------------------------------------------
    # get rn for backtest, based on HS300
    # --------------------------------------------------------------------
    # HS300 for ZZ800: E:\【Intern】\AlphaNet\OracleData\HS300Data\Merge Data\Shift1_
    # HS300 for AMarket: E:\【Intern】\AlphaNet\OracleData\HS300Data\Merge Data\
    # ZZ500 for ZZ800: E:\【Intern】\AlphaNet\OracleData\ZZ500Wind\Merge Data\Shift1_
    path = r'E:\【Intern】\AlphaNet\OracleData\AMarket\Merge Data\rn_TS_Y_3.npy'
    array_Y_3 = np.load(path, allow_pickle=True)[:, [0, 1, 2]]  # 对于 两个未来10天 [0, 1, 2]
    df_index = pd.DataFrame(array_Y_3, columns=['date', 'codes', 'next_rn'])
    df_index['next_rn'] = df_index['next_rn'].astype(np.float32)

    # ------------------------------------------------------------------------------------------
    # method 1: Predict as SyntaxFactor: pd.merge, 直接用输出作为合成因子
    # method 2: LSTM output as SyntaxFactor: np.concat, 中间过程作为因子，60因子。代码不完整
    # ------------------------------------------------------------------------------------------
    # method 1
    df_ec = pd.merge(df_index, df_basic, how='left', on=['date', 'codes'])
    df_ec = df_ec.dropna()
    print(df_ec.isna().sum())
    print(df_ec.dtypes)

    # method 2: 具体包括10个随机种子，每个随机种子下对于每支股票，每个时刻都有60个因子。
    # li_WindData_Y 只包含时间、股票
    # len_ = factor.shape[0]
    # df_ec = pd.DataFrame(np.concatenate([factor, li_WindData_Y], axis=1))
    # df_ec = df_ec.rename(columns={df_ec.columns[-3]: 'date', df_ec.columns[-2]: 'codes', df_ec.columns[-1]: 'next_rn'})
    # df_ec.iloc[:, range(0, 60)] = df_ec.iloc[:, range(0, 60)].astype(np.float32)

    # 行业、市值中性化
    path_industry = r'E:\【Intern】\AlphaNet\JQdict\AMarketIndustry\NewIndustry_AMarket.feather'
    df_industry = pd.read_feather(path_industry)  # , usecols=[0, 1, 4, 5]
    df_industry.date = df_industry.date.astype(str)
    df_ec = pd.merge(df_ec, df_industry, how='left', on=['date', 'codes'])
    df_ec = df_ec.dropna(axis=0, how='any', subset=['INDUSTRY_CODE', 'market_cap']).set_index(['date', 'codes'])
    df_ec = bl.neutralization(df_ec).reset_index().set_index(['date'])
    print(df_ec.isna().sum())
    print(df_ec.dtypes)

    # factor --> IC
    # 对因子进行行业、市值中性化，计算IC
    factor_col = [x for x in df_ec.columns if x not in ['date', 'codes', 'next_rn', 'INDUSTRY_CODE', 'market_cap']]
    df_IC = df_ec.groupby('date').apply(lambda x: [st.spearmanr(x[factor], x['next_rn'])[0] for factor in factor_col])
    df_IC = pd.DataFrame(df_IC.tolist(), index=df_IC.index, columns=factor_col)
    print(df_IC.describe())
    print('df_IC > 0 占比:', len(df_IC.query('ComFac >=0')) / len(df_IC))

    # today = datetime.date.today()
    # df_IC.to_csv(f'./IC/IC_wind{today}.csv')
    rankIC = 'One'  # 'rank', 'mean', 'One'
    if rankIC == 'rank':
        combine_df = bl.rankWeightIC(df_IC, df_IC.index.unique(), df_ec, columns=range(0, 60))
    elif rankIC == 'mean':
        array_EqaulWeightICfac = df_ec.loc[:, range(0, 60)].mean(axis=1).values.reshape(-1, 1)
        combine_df = pd.DataFrame(array_EqaulWeightICfac, columns=['comb_fac'], index=df_ec.index)
    else:
        combine_df = pd.DataFrame(df_ec.ComFac.values, columns=['comb_fac'], index=df_ec.index)

    combine_df[['codes', 'next_rn']] = df_ec.loc[:, ['codes', 'next_rn']]
    combine_df['next_rn'] = combine_df['next_rn'].astype(float)
    combine_df = combine_df.reset_index()
    print(combine_df.isna().sum())

    # 分组策略
    group = 10
    df_ec_ = bl.get_ec(combine_df, fac='comb_fac', ret_next='next_rn', group=group)
    plot_df, _, group_ = bl.plot_ec(df_ec_, ret_next='next_rn', fac='comb_fac')
    plot_df = plot_df - 1
    save_path = r'./figs/NewestSyntheticFactorV3self.png'
    fig_title = fr'AlphaNet-v3 group performance | Synthetic Factor of A Market | {group} Groups'
    bl.plot_performance(plot_df, fig_title, save_path)

    # df_ec.to_csv(f'./backtest/df_ec_de_industry_MV{today}.csv')
    # df_ec_.to_csv(f'./backtest/df_ec_groupStrategy{today}.csv')

    # 构建策略与大盘基准
    LastDate = '20220824'  # '20211227'， '20220824'
    path_index = r'E:\【Intern】\AlphaNet\TushareData\zz500New.csv'
    group_ = bl.transform_group(group_, group)
    top_strategy = df_ec_.query('group==@group_').reset_index(drop=True)
    df_hs300 = pd.read_csv(path_index, index_col=[0], usecols=[0, 1, 2, 3], dtype={'trade_date': str})
    df_hs300_bet10days = bl.align_index(df_hs300, df_ec, LastDate)
    df_hs300_bet10days = df_hs300_bet10days.rename(columns={'trade_date': 'date', 'rn': 'next_rn'})

    columns = ['Strategy', 'HS300']
    hs300DiffStrategy, plot_diff_rn = bl.diffIndex(top_strategy, df_hs300_bet10days)
    hs300DiffStrategyIndex, _ = bl.diffIndex(df_hs300_bet10days, df_hs300_bet10days)
    result, drawdown = bl.get_performance(hs300DiffStrategy, columns=columns, benchmark_col='HS300', Days=10)
    df_result_Y = bl.gey_Yperformance(hs300DiffStrategy, columns=columns, benchmark_col='HS300', Days=10)
    df_result_YINdex = bl.gey_Yperformance(hs300DiffStrategyIndex, columns=columns, benchmark_col='HS300', Days=10)
    print(result)
    print(df_result_Y)
    print(df_result_YINdex)   # 基准收益

    plot_diff_rn['maxdraw'] = drawdown['Strategy']
    save_path = r'./figs/NewAlphaNet_Strategy_HS300self.png'
    fig_title = r'AlphaNet Strategy (A Market) & HS300 | Top Strategy Performance'
    bl.plot_performance(plot_diff_rn, fig_title, save_path)

    # industry, 获取每年行业sum Top5
    path = 'E:\【Intern】\AlphaNet\JQdict'
    df_top_strategy = combine_df.query('group==@group_')
    df_top_strategy = pd.merge(df_top_strategy, df_industry, how='left', on=['date', 'codes'])
    df_top_strategy['INDUSTRY_CODE'] = df_top_strategy['INDUSTRY_CODE'].astype(int)
    # sw 一级行业分类
    df_sw = pd.read_excel(fr'{path}\SW1l.xlsx', usecols=[0, 1], dtype={0: 'int64'})
    df_sw.columns = ['INDUSTRY_CODE', 'INDUSTRY_NM']
    df_top_strategy = pd.merge(df_top_strategy, df_sw, how='left', on=['INDUSTRY_CODE'])
    df_top_industry = df_top_strategy.groupby(['date', 'INDUSTRY_NM']).size().reset_index().rename(columns={0: 'count'})
    # 行业sum top5
    df_top5_industry = df_top_industry.sort_values(by=['date', 'count'], ascending=[True, False]).groupby(['date']).head(5).reset_index(drop=True)
    df_top5_industry.date = pd.to_datetime(df_top5_industry.date)
    df_top5_industry = df_top5_industry.set_index(['date'])
    df_top5_industry = df_top5_industry.groupby([pd.Grouper(level=0, freq='Y'), 'INDUSTRY_NM']).sum().reset_index()
    df_top5_industry = df_top5_industry.sort_values(by=['date', 'count'], ascending=[1, 0]).groupby(['date']).head(5).reset_index(drop=True)

