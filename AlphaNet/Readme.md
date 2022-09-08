需要自己切换数据，处理输入数据的格式，否则是跑不通的，在此仅为记录，不为程序的完整运行负责。
具体运行包括以下两个步骤：
- 模型运行：主文件（`main_alpha`）
- 策略回测：主文件（`combine_ic.py`）

### 数据来源

- wind oracle 数据， 我code运行的数据不是这里上传的。 
- 聚宽数据，使用聚宽获取一些去ST，上市已满12个月的股票数据，然后和wind oracle的数据去merge

### 文件

预测值作为合成因子，Model的y是经过标准化处理之后的，同时也可以尝试其他处理方法； `model.py` 主要来源知乎的一篇文章。

- `model.py` 为比较原始的版本，可以run
- `modelV3.py` 为修改的V3 版本，算子有些许问题，使用GPU跑会快很快！
- `tools.py` 是从 **tushare** 获取数据，由于太慢，有限制，后期弃用。 `tools.py`有数据对齐函数，可以将聚宽中的数据格式与wind数据中的**date** 和 **codes** 对齐。
- `backtools.py` 为回测工具文件，包含单组分层IC测试，再次感谢东哥。
- `combine_ic.py` 为合成因子IC回测主文件。
- **data** 文件夹的数据与[2]相同

### 回测

- `main_alpha` 每个随机种子保存到本地的结果就是合成因子，需要将多个随机种子mean或者rank加权，生成最终的合成因子。
  - 单因子分层测试/每年回测 
  - Barr 组合优化，指数增强（不会做）


### 优化

模型没有进行组合优化，华泰后面写了一篇  **《华泰证券人工智能52：神经网络组合优化初探》**

### 参考资料

[1] [github: AlphaNetV3](https://github.com/Congyuwang/AlphaNetV3), 使用TensorFlow写的，参考了其中的code逻辑，没有运行

[2]  [知乎：AlphaNet因子挖掘网络——运算符嵌套和卷积神经网络](https://zhuanlan.zhihu.com/p/546110583)

[3]  [cnblogs: 股票因子挖掘神经网络构建](https://blog.csdn.net/qq_45137571/article/details/118532260) 构建网络结构；如果单纯用cpu跑，数据多了就很慢，可以继续优化

[4]  [cvxgrp-cvxpylayers](https://www.mianshigee.com/project/cvxgrp-cvxpylayers)
