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

# 模型左边的菜单选项
# 两个空格开头代表不可选，为Title
mainleftFrameTextList = ["  模型中心", "配置模型", "训练模型", "校验模型", "预测结果",
                         "  数据中心", "数据集管理", "数据分析"]

# 每种模型下有哪些模型
modelTypes2models = {
    "回归模型": [
        {
            "model_name": ("改进的自适应Lasso", "SALP"),
            "model_parameters": {
                "alpha": ("惩罚项系数", 0.001),
                "excludeVariablePercent": ("参数淘汰率", 0.25)
            }
        },
        {
            "model_name": ("多层线性模型", "HLM"),
            "model_parameters": {
                "parameter1": ("参数1", 0.001),
                "parameter2": ("参数2", 0.25),
                "parameter3": ("参数3", 123)
            }
        }],
    "决策分析": [
        {
            "model_name": ("层次分析法", "AHP")
        }],
    "参数寻优": [
        {
            "model_name": ("遗传算法", "GA")
        }],
}
