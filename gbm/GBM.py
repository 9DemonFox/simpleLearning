from model import Model
from sklearn.ensemble import GradientBoostingRegressor


class GBMModel(Model):
    def __init__(self, **kwargs):
        """
        n_estimators: int 也就是弱学习器的最大迭代次数，或者说最大的弱学习器的个数。
        一般来说n_estimators太小，容易欠拟合，n_estimators太大，又容易过拟合，一般选择一个适中的数值。默认是100。
        在实际调参的过程中，我们常常将n_estimators和下面介绍的参数learning_rate一起考虑。

        learning_rate: float 即每个弱学习器的权重缩减系数ν，也称作步长。迭代公式为fk(x)=fk−1(x)+νhk(x)。
        ν的取值范围为0<ν≤1。一般来说，可以从一个小一点的ν开始调参，默认是1。

        min_samples_split: int or float, default=2 分割一个内部节点所需的最小样本数：
        如果是int，则考虑min_samples_split作为最小值。
        如果是float，那么min_samples_split是一个分数，而ceil(min_samples_split * n_samples)是每个分割的最小样本数。

        max_depth: int, default=3, 单个回归估计量的最大深度。最大深度限制了树中的节点数。优化此参数以获得最佳性能。

　　　　 loss: 有均方差"ls", 绝对损失"lad", Huber损失"huber"和分位数损失“quantile”。默认是均方差"ls"。
          一般来说，如果数据的噪音点不多，用默认的均方差"ls"比较好。如果是噪音点较多，则推荐用抗噪音的损失函数"huber"。
          而如果我们需要对训练集进行分段预测的时候，则采用“quantile”。

        """
        self.model = GradientBoostingRegressor(**kwargs)

    def fit(self, trainX, trainY):
        self.model.fit(trainX, trainY)

    def predict(self, predictX):
        return self.model.predict(predictX)
