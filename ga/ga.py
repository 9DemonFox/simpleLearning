import numpy as np
import sys
sys.path.append('..')
from model import Model
from data.ga.dataLoder import GADataLoader


def get_fitness(c, pop, F, n, precisions, xmax, xmin, POP_SIZE):
    """

    Returns
    -------
    一维向量
        计算适应度.

    """
    x = translateDNA(pop, n, xmax, xmin, precisions, POP_SIZE)
    pred = F(x)
    if c == 1:
        return (pred - np.min(pred)) + 1e-10
    if c == 0:
        return -(pred - np.max(pred)) + 1e-10


def translateDNA(pop, n, xmax, xmin, precisions, POP_SIZE):
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
    x_pop = np.ones((n, POP_SIZE, precisions))
    x = np.ones((n, POP_SIZE))
    for i in range(n):
        x_pop[i] = np.array(pop[:, i::n])

    for i in range(n):
        x[i] = x_pop[i].dot(2 ** np.arange(precisions)[::-1]) / float(2 ** precisions - 1) * (
                xmax - xmin) + xmin
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


def get_fitness1(pop, F, n, precisions, xmax, xmin, POP_SIZE):
    """

    Returns
    -------
    pred : array
        种群所对应函数值.

    """
    x = translateDNA(pop, n, xmax, xmin, precisions, POP_SIZE)
    pred = F(x)
    return pred


def print_info(c, pop, F, n, precisions, xmax, xmin, POP_SIZE):
    """

    得到最优解

    """

    fitness = get_fitness1(pop, F, n, precisions, xmax, xmin, POP_SIZE)
    if c == 1:
        best_fitness_index = np.argmax(fitness)
    if c == 0:
        best_fitness_index = np.argmin(fitness)
    # print("optimal_value:", fitness[best_fitness_index])
    x = translateDNA(pop, n, xmax, xmin, precisions, POP_SIZE)
    return fitness[best_fitness_index], x[:, best_fitness_index:best_fitness_index + 1]
    # for i in range(n):
    # print(x[i][best_fitness_index])
    # print("最优的基因型：", pop[best_fitness_index])


# print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))


def ga(c, F, n, xmax, xmin, precisions, N_GENERATIONS, POP_SIZE, MUTATION_RATE, CROSSOVER_RATE):
    pop = np.random.randint(n, size=(POP_SIZE, precisions * n))
    for _ in range(N_GENERATIONS):
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE, POP_SIZE, precisions, MUTATION_RATE))

        fitness = get_fitness(c, pop, F, n, precisions, xmax, xmin, POP_SIZE)
        pop = select(pop, fitness, POP_SIZE)

    return print_info(c, pop, F, n, precisions, xmax, xmin, POP_SIZE)


class GAModel(Model):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        c:int,必选
        指定求函数的最大值或最小值，‘1’为求最大值，‘0’为求最小值
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
        self.c = kwargs["c"]
        self.n = kwargs["n"]
        if "xmax" not in kwargs.keys():
            self.xmax = 1000
        else:
            self.xmax = kwargs["xmax"]
        if "xmin" not in kwargs.keys():
            self.xmin = -1000
        else:
            self.xmin = kwargs["xmin"]
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
    
    def predictForUI(self, **kwargs):
        """ 返回结果到前端
        :return:
        """
        self.predictX = kwargs["predictX"]
        # 返回结果为字典形式
        y,x = self.predict()
        returnDic = {
            "最值": str(y),
            "变量取值": str(x)
        }
        return returnDic
    
       
    def fit(self):
        pass

    def predict(self, **kwargs):
        if "predictX" not in kwargs.keys():
            self.predictX = self.predictX
        else:
            self.predictX = kwargs["predictX"]
        return ga(self.c, self.predictX, self.n, self.xmax, self.xmin, self.precisions, self.N_GENERATIONS, self.POP_SIZE, self.MUTATION_RATE, self.CROSSOVER_RATE)



if __name__ == '__main__':


    c = 0
    a = GADataLoader()
    F = a.loadPredictData(predict_path="../data/ga/F.txt")
    n = 3
    xmax = 10
    xmin = -10
    precisions = 24
    N_GENERATIONS = 50
    POP_SIZE = 200
    MUTATION_RATE = 0.005
    CROSSOVER_RATE = 0.8
    model = GAModel(c=c, n=n, xmax=xmax, xmin=xmin, precisions=precisions, N_GENERATIONS=N_GENERATIONS, 
                    POP_SIZE=POP_SIZE,MUTATION_RATE=MUTATION_RATE, CROSSOVER_RATE=CROSSOVER_RATE)
    print(model.predictForUI(predictX=F))
    
