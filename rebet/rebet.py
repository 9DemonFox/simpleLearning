import numpy as np
from sklearn import tree
from data.rebet.dataLoder import REBETDataLoader
from model import Model
from scipy.stats import gamma
import math
from scipy import stats
from sklearn.metrics import mean_squared_error


def update(trainX, trainY, epoch, n, N, m, D, q, u, σ2, clf, y0, z, M):
    """
    

    Parameters
    ----------   
    epoch:int
        循环轮数
    n : int
        观测对象种类数量
    N : int
        观测次数
    m : int
        每种观测对象的观测次数
    D:  array
        迪利克雷参数中的协方差矩阵
    q : int
        随机效应变量数
    u : array
        随机效应变量
    σ2 : float
        误差的方差
    clf : tree
        决策回归树
    y0 : array
    z : array
        随机效应参数
    M:  float
        迪利克雷分布参数

    Returns
    -------
    None.

    """
    u[0] = np.random.multivariate_normal(np.zeros(q), D, size=(1)).reshape(q, 1)
    #print(u)
    for i in range(n):
        if i != 0:
            j = i + 1
            a = np.random.binomial(1, M / (M + j - 1), 1)
            if a == 1:
                u[i] = np.random.multivariate_normal(np.zeros(q), D, size=(1)).reshape(q, 1)

            else:
                b = np.random.randint(1, j)
                u[i] = u[b - 1]

    for i in range(epoch):
        for j in range(n):
            y0[j * m:(j + 1) * m, 0:1] = trainY[j * m:(j + 1) * m, 0:1] - np.dot(z[j], u[j])
        clf.fit(trainX, y0)
        y1 = clf.predict(trainX).reshape(N, 1)

        a = 0
        for j in range(n):
            #print(y1[j * m:(j + 1) * m, 0:1] , np.dot(z[j], u[j]))
            a = a + np.dot((trainY[j * m:(j + 1) * m, 0:1] - y1[j * m:(j + 1) * m, 0:1] - np.dot(z[j], u[j])).T,
                           (trainY[j * m:(j + 1) * m, 0:1] - y1[j * m:(j + 1) * m, 0:1] - np.dot(z[j], u[j])))
         

        a = a / 2
        #print(a)
        b = N / 2 + 1
        if (a==0):
            σ2 = 0.0000001
        else:
            σ2 = 1 / gamma.rvs(b, scale=1 / a)
        if (σ2<=0.0000001):
            σ2 = 0.0000001
        #print(σ2)
        r = np.empty((n, q, 1))
        k = 0

        l = np.zeros(n)
        c = 0
        for t in range(n):
            f = stats.multivariate_normal.pdf((trainY[t * m:(t + 1) * m, 0:1]).reshape(m),
                                              mean=(y1[t * m:(t + 1) * m, 0:1] + np.dot(z[t], u[t])).reshape(m),
                                              cov=σ2 * np.identity(m))
            c = c + stats.multivariate_normal.pdf((trainY[t * m:(t + 1) * m, 0:1]).reshape(m),
                                                  mean=(y1[t * m:(t + 1) * m, 0:1] + np.dot(z[t], u[t])).reshape(m),
                                                  cov=σ2 * np.identity(m))
            l[t] = f

        for j in range(n):
            a = np.identity(q)
            d = np.identity(q)
            Q = 1/(np.dot(z[j].T, z[j]) / σ2 + 1/D)

            U = np.dot(np.dot(z[j], Q), z[j].T) / σ2 - np.identity(m)
            #print(Q,D)
            I = M * math.pow(σ2 * 2 * math.pi, m / (-2)) * math.pow(np.abs(D), -1 / 2) * math.pow(np.abs(Q), 1 / 2) * math.exp((np.dot(
                np.dot((trainY[j * m:(j + 1) * m, 0:1] - y1[j * m:(j + 1) * m, 0:1]).T, U),
                trainY[j * m:(j + 1) * m, 0:1] - y1[j * m:(j + 1) * m, 0:1])) / (2 * σ2))

            w = I
            f = c - l[j]
            p = w / (w + f)
            if np.isnan(p) or p < 0:
                p = 0
            if np.isinf(p) or p > 1:
                p = 1
            a = np.random.binomial(1, p, 1)
            if a == 1:
                d = np.identity(q)
                mean = np.dot(np.dot(
                    np.linalg.inv(np.dot(z[j].T, z[j]) + σ2 * np.divide(d, D, out=np.zeros_like(d), where=D != 0)),
                    z[j].T), trainY[j * m:(j + 1) * m, 0:1] - y1[j * m:(j + 1) * m, 0:1])
                d = np.identity(q)
                cov = σ2 * np.linalg.inv(
                    np.dot(z[j].T, z[j]) + σ2 * np.divide(d, D, out=np.zeros_like(d), where=D != 0))
                u[j] = np.random.multivariate_normal(mean.reshape(q), cov, size=(1)).reshape(q, 1)
                #print(mean)
                r[k] = u[j]
                k = k + 1
            else:
                s = np.delete(l, j)
                b = np.random.choice(np.arange(n - 1), size=1, replace=True,
                                     p=(s) / (s.sum()))

                if b < j:
                    u[j] = u[b]
                else:
                    u[j] = u[b + 1]
                a = 0
                for t in range(k):
                    if (r[t] == u[j]).all():
                        a = 1
                        break
                if a == 0:
                    r[k] = u[j]
                    k = k + 1
        r = r[0:k, :, :].reshape(k, q)

        a = np.dot(r.T, r) / 2
        b = k / 2 + 1
        D = 1 / gamma.rvs(b, scale=np.abs(1 / a))
        #print(D)
        if (D<math.pow(10,-10)):
            D = np.identity(q)
            u[0] = np.random.multivariate_normal(np.zeros(q), D, size=(1)).reshape(q, 1)
    #print(u)
            for i in range(n):
                if i != 0:
                    j = i + 1
                    a = np.random.binomial(1, M / (M + j - 1), 1)
                    if a == 1:
                        u[i] = np.random.multivariate_normal(np.zeros(q), D, size=(1)).reshape(q, 1)
                    else:
                        b = np.random.randint(1, j)
                        u[i] = u[b - 1]
    return D,σ2


class REBETModel(Model):

    def __init__(self, **kwargs):
        """
        
        n:int,可选(默认 = 100)
        观测对象种类数量
        q：int
        随机效应变量数
        D:array
        迪利克雷参数中的协方差矩阵
        u:array
        随机效应变量
        σ2:float
        误差的方差
        M:float,可选(默认 = 10)
        迪利克雷分布参数
        epoch:int,可选(默认=50)
        循环轮数
        k:int,可选(默认=1)
        表示第k个变量作为随机效应变量

        """
        if "n" not in kwargs.keys():
            self.n = 1
        else:
            self.n = kwargs["n"]
        if "M" not in kwargs.keys():
            self.M = 10
        else:
            self.M = kwargs["M"]
        if "epoch" not in kwargs.keys():
            self.epoch = 50
        else:
            self.epoch = kwargs["epoch"]

        if "k" not in kwargs.keys():
            self.k = 1
        else:
            self.k = kwargs["k"]
        if "ga" not in kwargs.keys():
            self.ga = 0
        else:
            self.ga = kwargs["ga"]
        self.q = 1
        self.D = np.identity(self.q)
        self.u = np.zeros((self.n, 1,1))
        self.σ2 = 1.0
        
    def fitForUI(self, **kwargs):
        """ 返回结果到前端
        :return:
        """
        # 返回结果为字典形式
        self.fit(**kwargs)
        returnDic = {
            "None": "None",
        }
        return returnDic
    
    def testForUI(self, **kwargs):
        """
        :param kwargs:
        :return: 字典形式结果
        """
        if (self.ga==0):  
            mse = self.test(**kwargs)
            returnDic = {
              "提示":"未使用参数寻优",
              "mean_squared_error": str(mse)
            }
        if (self.ga==1):    
            x,mse = self.test(**kwargs)
            returnDic = {
              "提示":"使用参数寻优，显示寻优参数值",
              "epoch": str(x[0][0].astype(int)),
              "M": str(x[2][0]),
              "k": str(x[1][0].astype(int)),
              "mean_squared_error": str(mse)
            }
        return returnDic
    
    def predictForUI(self, **kwargs):
        """
        :param kwargs:
        :return: 字典形式结果
        """
        returnDic = {
            "预测结果": None
        }
        predictResult = self.predict(**kwargs)
        returnDic["预测结果"] = str(predictResult)
        return returnDic
    
    def fit(self, **kwargs):

        """
        
        N:int
        观测次数
        m:int
        每种观测对象的观测次数
        z:array
        随机效应参数

        """

        self.trainX = kwargs["trainX"]
        self.trainY = kwargs["trainY"]
            
        self.clf = tree.DecisionTreeRegressor(criterion='mse', max_depth=(self.trainX.shape[1]))
        N = self.trainX.shape[0]
        self.trainY = self.trainY.reshape(N,1)
        m = int(N / self.n)
        z = np.zeros((self.n, m, self.q))
        for i in range(self.n):
            z[i:i + 1, :, 0:1] = z[i:i + 1, :, 0:1]  + self.trainX[m * i:m * (i + 1), self.k-1:self.k]
        #print(z)
        y0 = np.empty((N, 1))
        update(self.trainX, self.trainY, self.epoch, self.n, N, m, self.D, self.q, self.u, self.σ2, self.clf, y0, z, self.M)
        for j in range(self.n):
            y0[j * m:(j + 1) * m, 0:1] = self.trainY[j * m:(j + 1) * m, 0:1] - np.dot(z[j], self.u[j])

        self.clf2 = tree.DecisionTreeRegressor(criterion='mse', random_state=0, max_depth=(self.trainX.shape[1]))
        self.clf2.fit(self.trainX, y0)
        
        #print(z,self.u,np.dot(z[j], self.u[j]), self.clf2.predict(self.trainX),self.clf2.predict(self.trainX).reshape(N, 1)+np.dot(z[j], self.u[j]))
        
           

    def test(self, **kwargs):
        self.testX = kwargs["testX"]
        self.testY = kwargs["testY"]
        if (self.ga==1):
            N = 3
            m = [[50,200],[1,self.trainX.shape[1]],[0.01,100]]
            precisions = 24
            N_GENERATIONS = 50
            POP_SIZE = 50
            MUTATION_RATE = 0.005
            CROSSOVER_RATE = 0.8
            model1 = GAModel(n=self.n,testX=self.testX, testY=self.testY, trainX=self.trainX, trainY=self.trainY, N=N, m=m, precisions=precisions, N_GENERATIONS=N_GENERATIONS, 
                             POP_SIZE=POP_SIZE,MUTATION_RATE=MUTATION_RATE, CROSSOVER_RATE=CROSSOVER_RATE)
            y,x = model1.predict()
            self.model = REBETModel(n=self.n, epoch=x[0][0].astype(int), M=x[2][0], k=x[1][0].astype(int))
            self.model.fit(trainX=self.trainX, trainY=self.trainY)
            N = self.testX.shape[0]
            m = int(N / self.n)
            z = np.zeros((self.n, m, self.q))
            for i in range(self.n):
                z[i:i + 1, :, 0:1] = z[i:i + 1, :, 0:1]  + self.testX[m * i:m * (i + 1), self.k-1:self.k]
            x0 = self.clf2.predict(self.testX).reshape(N, 1)
            for j in range(self.n):
                x0[j * m:(j + 1) * m, 0:1] = x0[j * m:(j + 1) * m, 0:1] + np.dot(z[j], self.u[j])
            mse = mean_squared_error(x0, self.testY)
            return x,mse
        N = self.testX.shape[0]
        m = int(N / self.n)
        z = np.zeros((self.n, m, self.q))
        for i in range(self.n):
            z[i:i + 1, :, 0:1] = z[i:i + 1, :, 0:1]  + self.testX[m * i:m * (i + 1), self.k-1:self.k]
        x0 = self.clf2.predict(self.testX).reshape(N, 1)
        #print(x0,self.testX)
        for j in range(self.n):
            x0[j * m:(j + 1) * m, 0:1] = x0[j * m:(j + 1) * m, 0:1] + np.dot(z[j], self.u[j])
        #print(x0)
        return mean_squared_error(x0, self.testY)
    
    def predict(self, **kwargs):
        self.predictX = kwargs["predictX"]
        N = self.predictX.shape[0]
        m = int(N / self.n)
        z = np.zeros((self.n, m, self.q))
        for i in range(self.n):
            z[i:i + 1, :, 0:1] = z[i:i + 1, :, 0:1] + self.predictX[m * i:m * (i + 1), self.k-1:self.k]
        x0 = self.clf2.predict(self.predictX).reshape(N, 1)
        for j in range(self.n):
            x0[j * m:(j + 1) * m, 0:1] = x0[j * m:(j + 1) * m, 0:1] + np.dot(z[j], self.u[j])
        return x0



def get_fitness(n,trainX,trainY,testX,testY,pop, N, precisions, m, POP_SIZE):
    """

    Returns
    -------
    一维向量
        计算适应度.

    """
    x = translateDNA(pop, N, m, precisions, POP_SIZE)
    y1 = x[0].astype(int)
    y2 = x[1].astype(int)
    pred = np.zeros([POP_SIZE,])
    for i in range(POP_SIZE):    
        model = REBETModel(n=n, epoch=y1[i], M=x[2][i], k=y2[i], ga=0)
        model.fit(trainX=trainX, trainY=trainY)
        pred[i] = model.test(testX=testX, testY=testY)
    return -(pred - np.max(pred)) + 1e-10


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


def get_fitness1(n,trainX,trainY,testX,testY,pop, N, precisions, m, POP_SIZE):
    """

    Returns
    -------
    pred : array
        种群所对应函数值.

    """
    x = translateDNA(pop, N, m, precisions, POP_SIZE)
    pred = np.zeros([POP_SIZE,])
    y1 = x[0].astype(int)
    y2 = x[1].astype(int)
    for i in range(POP_SIZE):    
        model = REBETModel(n=n, epoch=y1[i], M=x[2][i], k=y2[i], ga=0)
        model.fit(trainX=trainX, trainY=trainY)
        pred[i] = model.test(testX=testX, testY=testY)
    return pred


def print_info(n,trainX,trainY,testX,testY,pop, N, precisions, m, POP_SIZE):
    """

    得到最优解

    """

    fitness = get_fitness1(n,trainX,trainY,testX,testY,pop, N, precisions, m, POP_SIZE)
    best_fitness_index = np.argmin(fitness)
    # print("optimal_value:", fitness[best_fitness_index])
    x = translateDNA(pop, N, m, precisions, POP_SIZE)
    return fitness[best_fitness_index], x[:, best_fitness_index:best_fitness_index + 1]
    # for i in range(n):
    # print(x[i][best_fitness_index])
    # print("最优的基因型：", pop[best_fitness_index])


# print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))


def ga(n,trainX,trainY,testX,testY, N, m, precisions, N_GENERATIONS, POP_SIZE, MUTATION_RATE, CROSSOVER_RATE):
    pop = np.random.randint(2, size=(POP_SIZE, precisions * N))
    for _ in range(N_GENERATIONS):
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE, POP_SIZE, precisions, MUTATION_RATE))

        fitness = get_fitness(n,trainX,trainY,testX,testY,pop, N, precisions, m, POP_SIZE)
        pop = select(pop, fitness, POP_SIZE)

    return print_info(n,trainX,trainY,testX,testY,pop, N, precisions, m, POP_SIZE)


class GAModel(Model):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        n:int,必选
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
        self.n = kwargs["n"]
        self.trainX = kwargs["trainX"]
        self.trainY = kwargs["trainY"]
        self.testX = kwargs["testX"]
        self.testY = kwargs["testY"]
        self.N = kwargs["N"]
        self.m = kwargs["m"]
        if "precisions" not in kwargs.keys():
            self.precisions = 24
        else:
            self.precisions = kwargs["precisions"]
        if "POP_SIZE" not in kwargs.keys():
            self.POP_SIZE = 200
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
    
    def predictForUI(self):
        """ 返回结果到前端
        :return:
        """
        # 返回结果为字典形式
        y,x = self.predict()
        returnDic = {
            "最值": str(y),
            "变量取值": str(x)
        }
        return returnDic
    
       
    def fit(self):
        pass

    def predict(self):
        return ga(self.n,self.trainX,self.trainY,self.testX,self.testY, self.N, self.m, self.precisions, self.N_GENERATIONS, self.POP_SIZE, self.MUTATION_RATE, self.CROSSOVER_RATE)

if __name__ == '__main__':
    n = 1
    epoch = 200
    k = 2
    M = 1
    datapath1="../data/rebet/data_train.xlsx"
    datapath2="../data/rebet/data_test.xlsx"
    datapath3="../data/rebet/data_predict.xlsx"
    dataloader = REBETDataLoader()
    trainX, trainY = dataloader.loadTrainData(train_path=datapath1)
    testX, testY = dataloader.loadTestData(test_path=datapath2)
    predictX = dataloader.loadPredictData(predict_path=datapath3)
    model = REBETModel(n=n, epoch=epoch, M=M, k=k ,ga=0)
    model.fitForUI(trainX=trainX, trainY=trainY)
    print(model.testForUI(testX=testX, testY=testY))
    print(model.predictForUI(predictX=predictX))
    
    