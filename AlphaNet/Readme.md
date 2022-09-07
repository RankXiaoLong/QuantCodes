参考 [股票因子挖掘神经网络构建](https://blog.csdn.net/qq_45137571/article/details/118532260) 构建网络结构；

### 数据来源

wind oracle 数据

### 文件

预测值作为合成因子，Model的y是经过标准化处理之后的，同时也可以尝试其他处理方法； `model.py` 主要来源知乎的一篇文章。


`modelV3.py` 为修改的V3 版本，算子有些许问题，使用GPU跑会快很快！

### 优化

模型没有进行组合优化，华泰后面谢了一篇神经网络组合优化cvxpylayers

### 参考资料

-  [github: AlphaNetV3](https://github.com/Congyuwang/AlphaNetV3), 使用TensorFlow写的，参考了其中的code逻辑，没有运行
-  [知乎：AlphaNet因子挖掘网络——运算符嵌套和卷积神经网络](https://zhuanlan.zhihu.com/p/546110583)
-  [cvxgrp-cvxpylayers](https://www.mianshigee.com/project/cvxgrp-cvxpylayers)
