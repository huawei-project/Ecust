# 说明
- 只包含多光谱数据
- 从每份多光谱数据(23通道, 550nm~990nm, 间隔20nm)中依次抽取训练集样本
- 将相邻的波段放入验证集，其余放入测试集
- 训练集样本个数设置为6，则验证集12左右，剩余5张放入测试集
- 最终`split_4`最终统计结果如下
    ```
    number of samples: 37625
    number of train:  9168, 
    number of valid: 17427, 
    number of test:  11030, 
    train: valid: test:  0.244: 0.463: 0.293
    ```
