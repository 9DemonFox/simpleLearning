from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import numpy as np
from data.gbm.dataLoader import GBMDataLoader
from model import Model


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
        self.ga = kwargs["ga"]
        self.n_estimators = kwargs["n_estimators"]
        self.max_depth = kwargs["max_depth"]
        self.learning_rate = kwargs["learning_rate"]
        self.loss = kwargs["loss"]
        params = {'n_estimators': self.n_estimators,
              'max_depth': self.max_depth,
              'min_samples_split': 5,
              'learning_rate': self.learning_rate,
              # 'loss': 'lad',
               'loss': self.loss}
        self.model = GradientBoostingRegressor(**params)
       

    def fit(self, trainX, trainY):   
        self.trainX = trainX
        self.trainY = trainY 
        self.model.fit(trainX, trainY)
        
    def test(self, testX, testY):
        if (self.ga==1):
            N = 3
            m = [[50,200],[1,5],[0.01,1]]
            precisions = 24
            N_GENERATIONS = 50
            POP_SIZE = 50
            MUTATION_RATE = 0.005
            CROSSOVER_RATE = 0.8
            model1 = GAModel(loss=self.loss,testX=testX, testY=testY, trainX=self.trainX, trainY=self.trainY, N=N, m=m, precisions=precisions, N_GENERATIONS=N_GENERATIONS, 
                             POP_SIZE=POP_SIZE,MUTATION_RATE=MUTATION_RATE, CROSSOVER_RATE=CROSSOVER_RATE)
            y,x = model1.predict()
            params = {'n_estimators': x[0][0].astype(int),
              'max_depth': x[1][0].astype(int),
              'min_samples_split': 5,
              'learning_rate': x[2][0],
              # 'loss': 'lad',
               'loss': self.loss}
            self.model = GradientBoostingRegressor(**params)
            self.model.fit(self.trainX, self.trainY)
            predictResult = self.model.predict(testX)
            mse = mean_squared_error(predictResult, testY)
            return x,mse
        predictResult = self.model.predict(testX)
        mse = mean_squared_error(predictResult, testY)
        return mse

    def predict(self, predictX):
        return self.model.predict(predictX)

    def fitForUI(self, **kwargs):
        assert "trainX" in kwargs.keys()
        assert "trainY" in kwargs.keys()
        trainX, trainY = kwargs.get("trainX"), kwargs.get("trainY")
        returnDic = {
            "None": "None",
        }
        self.fit(trainX, trainY)
        return returnDic

    def testForUI(self, **kwargs):

        assert "testX" in kwargs.keys()
        assert "testY" in kwargs.keys()

        testX, testY = kwargs.get("testX"), kwargs.get("testY")
        
        # returnDic["predict_result"] = str(predictResult)
        if (self.ga==0):  
            mse = self.test(testX, testY)
            returnDic = {
              "提示":"未使用参数寻优",
              "mean_squared_error": str(mse)
            }
        if (self.ga==1):    
            x,mse = self.test(testX, testY)
            returnDic = {
              "提示":"使用参数寻优，显示寻优参数值",
              "n_estimators": str(x[0][0].astype(int)),
              "max_depth": str(x[1][0].astype(int)),
              "learning_rate": str(x[2][0]),
              "mean_squared_error": str(mse)
            }
        return returnDic

    def predictForUI(self, **kwargs):
        returnDic = {
            "predict_result": None
        }
        assert "predictX" in kwargs.keys()
        predictX = kwargs.get("predictX")
        predictY = self.predict(predictX=predictX)
        returnDic["predict_result"] = str(predictY)
        return returnDic



def get_fitness(loss,trainX,trainY,testX,testY,pop, N, precisions, m, POP_SIZE):
    """

    Returns
    -------
    一维向量
        计算适应度.

    """
    x = translateDNA(pop, N, m, precisions, POP_SIZE)
    y1 = x[0].astype(int)
    y2 = x[1].astype(int)
    y3 = x[2]
    pred = np.zeros([POP_SIZE,])
    for i in range(POP_SIZE): 
        params = {'n_estimators': y1[i],
              'max_depth': y2[i],
              'min_samples_split': 5,
              'learning_rate': y3[i],
              # 'loss': 'lad',
              'loss': loss}
        model = GBMModel(**params,ga=0)
        model.fit(trainX=trainX, trainY=trainY)
        pred[i] = model.test(testX=testX, testY=testY)
    return -(pred - np.max(pred)) + 1e-28


def translateDNA(pop, N, m, precisions, POP_SIZE):
    """

    Parameters
    ----------
    pop : array
    pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    
    Returns
    -------
    x : array
        解码.

    """
    x_pop = np.ones((N, POP_SIZE, precisions))
    x = np.ones((N, POP_SIZE))
    for i in range(N):
        x_pop[i] = np.array(pop[:, i::N])

    for i in range(N):
        x[i] = x_pop[i].dot(2 ** np.arange(precisions)[::-1]) / float(2 ** precisions - 1) * (
                m[i][1] - m[i][0]) + m[i][0]
        if (i==0 or i==1):
            for j in range(POP_SIZE):
                x[i][j]=np.round(x[i][j])
    return x


def crossover_and_mutation(pop, CROSSOVER_RATE, POP_SIZE, precisions, MUTATION_RATE):
    """

    Returns
    -------
    new_pop : array
    对种群进行交叉与变异.

    """
    new_pop = []

    for father in pop:
        child = father
        if np.random.rand() < CROSSOVER_RATE:
            mother = pop[np.random.randint(POP_SIZE)]
            cross_points = np.random.randint(low=0, high=precisions * 2)
            child[cross_points:] = mother[cross_points:]
        mutation(child, MUTATION_RATE, precisions)
        new_pop.append(child)
    return new_pop


def mutation(child, MUTATION_RATE, precisions):
    """

    变异

    """
    if np.random.rand() < MUTATION_RATE:
        mutate_point = np.random.randint(0, precisions)
        child[mutate_point] = child[mutate_point] ^ 1


def select(pop, fitness, POP_SIZE):
    """

    Returns
    -------
    array
        重新选择种群.

    """
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness) / (fitness.sum()))
    return pop[idx]


def get_fitness1(loss,trainX,trainY,testX,testY,pop, N, precisions, m, POP_SIZE):
    """

    Returns
    -------
    pred : array
        种群所对应函数值.

    """
    x = translateDNA(pop, N, m, precisions, POP_SIZE)
    y1 = x[0].astype(int)
    y2 = x[1].astype(int)
    pred = np.zeros([POP_SIZE,])
    for i in range(POP_SIZE): 
        params = {'n_estimators': y1[i],
              'max_depth': y2[i],
              'min_samples_split': 5,
              'learning_rate': x[2][i],
              # 'loss': 'lad',
              'loss': loss}
        model = GBMModel(**params,ga=0)
        model.fit(trainX=trainX, trainY=trainY)
        pred[i] = model.test(testX=testX, testY=testY)
    return pred


def print_info(loss,trainX,trainY,testX,testY,pop, N, precisions, m, POP_SIZE):
    """

    得到最优解

    """

    fitness = get_fitness1(loss,trainX,trainY,testX,testY,pop, N, precisions, m, POP_SIZE)
    best_fitness_index = np.argmin(fitness)
    # print("optimal_value:", fitness[best_fitness_index])
    x = translateDNA(pop, N, m, precisions, POP_SIZE)
    return fitness[best_fitness_index], x[:, best_fitness_index:best_fitness_index + 1]
    # for i in range(n):
    # print(x[i][best_fitness_index])
    # print("最优的基因型：", pop[best_fitness_index])


# print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))


def ga(loss,trainX,trainY,testX,testY, N, m, precisions, N_GENERATIONS, POP_SIZE, MUTATION_RATE, CROSSOVER_RATE):
    pop = np.random.randint(2, size=(POP_SIZE, precisions * N))
    for _ in range(N_GENERATIONS):
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE, POP_SIZE, precisions, MUTATION_RATE))

        fitness = get_fitness(loss,trainX,trainY,testX,testY,pop, N, precisions, m, POP_SIZE)
        pop = select(pop, fitness, POP_SIZE)

    return print_info(loss,trainX,trainY,testX,testY,pop, N, precisions, m, POP_SIZE)


class GAModel():
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        N:int,必选
        指定函数所含变量个数
        ranges:array,必选
        指定各个变量的取值范围
        precisions:int,可选（默认值 = 24）
        指定精度
        N_GENERATIONS:int,可选（默认值 = 50）
        指定迭代轮数
        POP_SIZE:int,可选（默认值 = 200）
        指定种群大小
        MUTATION_RATE:float,可选（默认值 = 0.005）
        指定变异概率
        CROSSOVER_RATE:float,可选（默认值 = 0.8）
        指定交叉概率
        """
        self.trainX = kwargs["trainX"]
        self.trainY = kwargs["trainY"]
        self.testX = kwargs["testX"]
        self.testY = kwargs["testY"]
        self.N = kwargs["N"]
        self.m = kwargs["m"]
        self.loss = kwargs["loss"]
        if "precisions" not in kwargs.keys():
            self.precisions = 24
        else:
            self.precisions = kwargs["precisions"]
        if "POP_SIZE" not in kwargs.keys():
            self.POP_SIZE = 50
        else:
            self.POP_SIZE = kwargs["POP_SIZE"]
        if "MUTATION_RATE" not in kwargs.keys():
            self.MUTATION_RATE = 0.005
        else:
            self.MUTATION_RATE = kwargs["MUTATION_RATE"]
        if "CROSSOVER_RATE" not in kwargs.keys():
            self.CROSSOVER_RATE = 0.8
        else:
            self.CROSSOVER_RATE = kwargs["CROSSOVER_RATE"]
        if "N_GENERATIONS" not in kwargs.keys():
            self.N_GENERATIONS = 50
        else:
            self.N_GENERATIONS = kwargs["N_GENERATIONS"]

    def predict(self):
        return ga(self.loss,self.trainX,self.trainY,self.testX,self.testY, self.N, self.m, self.precisions, self.N_GENERATIONS, self.POP_SIZE, self.MUTATION_RATE, self.CROSSOVER_RATE)


if __name__ == "__main__":
    # Quadric 损失函数 (y-f(x))^2 / 2 -> ls
    # Laplace 损失函数 abs(y-f(x)) -> lad
    params = {'n_estimators': 195,
              'max_depth': 5,
              'min_samples_split': 5,
              'learning_rate': 0.76,
              # 'loss': 'lad',
              'loss': 'ls'}
    gbm_reg = GBMModel(**params,ga=1)
    train_path = "../data/gbm/gbm_train_data.xlsx"
    test_path = "../data/gbm/gbm_test_data.xlsx"
    predict_path = "../data/gbm/gbm_predict_data.xlsx"
    gbm_loader = GBMDataLoader()

    trainX, trainY = gbm_loader.loadTrainData(train_path=train_path)
    print(trainX.shape, "\n", trainY.shape)
    testX, testY = gbm_loader.loadTestData(test_path=test_path)
    print(testX.shape, "\n", testY.shape)
    predictX = gbm_loader.loadPredictData(predict_path=predict_path)
    print(predictX.shape)

    gbm_reg.fitForUI(trainX=trainX, trainY=trainY)
    predict_result = gbm_reg.testForUI(testX=testX, testY=testY)
    print(predict_result)
    predictY = gbm_reg.predictForUI(predictX=predictX)
    print(predictY)
