# import datetime
# import math
# import time
# from datetime import timedelta
# import pandas as pd
# import statsmodels.api as sm
# import matplotlib
# from matplotlib import pyplot as plt
#
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from torch import optim
# import torch.nn.functional as F
# import tushare as ts

import numpy as np
import tools as tl

if __name__ == '__main__':

    X_train, y_train = tl.get_dataset('20190901', '20191231', 10, 30, 10)
    X_test, y_test = tl.get_dataset('20200101', '20200630', 10, 30, 10)

    print("there are in total", len(X_train), "training samples")
    print("there are in total", len(X_test), "testing samples")

    X_array = np.array(X_train)
    y_array = np.array(y_train)
    Xe = np.array(X_test)
    ye = np.array(y_test)
    np.save('./Data/X_train.npy', X_array)
    np.save('./Data/y_train.npy', y_array)
    np.save('./Data/X_test.npy', Xe)
    np.save('./Data/y_test.npy', ye)
