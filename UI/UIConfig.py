# 模型选择 (英文缩写, 全称)
machineLearningModels = [("AHP", "层次分析法"),
                         ("SALP", "基于贝叶斯和偏最小二乘的Lasso"),
                         ("结合回归树与EM算法的混合效应模型", "MERT"),
                         ("结合回归树与迪利克雷过程贝叶斯分析的混合效应模型", "REBET"),
                         ("遗传算法", "GA"),
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
                "alpha": ("惩罚项系数", 0.1),
                "max_iter": ("最大迭代次数", 500),
                "ex_var_per": ("参数淘汰率", 0.25),
                "k": ("重构样本数量", 10)
            }
        },
        {
            "model_name": ("梯度提升机", "GBM"),
            "model_parameters": {
                "n_estimators": ("弱学习器的个数", 100),
                "learning_rate": ("学习率", 1),
                "max_depth": ("单个回归估计量的最大深度", 3),
                "loss": ("损失函数", "\"ls\"")
            }
        },
        {
            "model_name": ("多层线性模型", "HLM"),
            "model_parameters": {
                "No Parameter": ("无参数", "None"),
            }
        },
        {
            "model_name": ("回归树与EM算法的混合效应模型", "MERT"),
            "model_parameters": {
                "n": ("观测对象种类数量",100),
                "epoch": ("迭代轮数",50),
                "k": ("指定第k个变量作为随机效应变量",1)
            }
        },
        {
            "model_name": ("回归树与Dirichlet贝叶斯分析的混合效应模型", "REBET"),
            "model_parameters": {
                "n": ("观测对象种类数量",100),
                "epoch": ("迭代轮数",50),
                "M": ("迪利克雷过程离散程度",10),
                "k": ("指定第k个变量作为随机效应变量",1)
            }
        }],
    "决策分析": [
        {
            "model_name": ("层次分析法", "AHP"),
            "model_parameters": {
                "No Parameter": ("无参数", "None"),
            }
        }],
    "参数寻优": [
        {
            "model_name": ("遗传算法", "GA"),
            "model_parameters": {
                "c": ("指定求函数的最大值或最小值",0),
                "n": ("函数所含变量个数",1),
                "xmin": ("所有变量的上界",1000),
                "xmax": ("所有变量的下界",-1000),
                "precisions": ("精度",24),
                "N_GENERATIONS": ("迭代轮数",50),
                "POP_SIZE": ("种群大小",200),
                "MUTATION_RATE": ("变异概率",0.005),
                "CROSSOVER_RATE": ("交叉概率",0.8),
            }
        }],
}
