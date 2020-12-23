# 模型选择 (英文缩写, 全称)
machineLearningModels = [("AHP", "层次分析法"),
                         ("SALP", "基于贝叶斯和偏最小二乘的Lasso")
                         ]

# 模型默认参数 参数名 默认值
modelDefaultConfig = {
    "AHP": {
        "模型全称": "层次分析法",
        "模型说明": "决策分析",
        "数据说明": "比较矩阵配置文件 参考doc/ML算法.doc 第7节",
        "input: 比较矩阵配置文件": "../data/ahp/ahpInput.txt",
        "out 各维度权重": "text"
    },
    "SALP": {
        "模型说明": "拟合数据 y = βX",
        "数据说明": "x输入为：数据集数目*特征数目，y输入形状 = 数据集数目",
        "input: 数据集文件": "../data/dalp/SALP_DATA.npy",
        "parameter": {
            "alpha": 0.5,
        },
        "out 测试集平均差": "",
        "out 预测结果": "",
    }
}
