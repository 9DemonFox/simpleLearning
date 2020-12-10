
import numpy as np
from model import Model

def get_fitness(c,pop,F,n,precisions,ranges,POP_SIZE): 
    """

    Returns
    -------
    一维向量
        计算适应度.

    """
    x = translateDNA(pop,n,ranges,precisions,POP_SIZE)
    pred = F(x)
    if c==1:
        return (pred - np.min(pred)) + 1e-3 #减去最小的适应度是为了防止适应度出现负数，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)],最后在加上一个很小的数防止出现为0的适应度
    if c==0:
        return -(pred - np.max(pred)) + 1e-3


def translateDNA(pop,n,ranges,precisions,POP_SIZE):
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
    x_pop = np.ones((n,POP_SIZE,precisions))
    x = np.ones((n,POP_SIZE))
    for i in range(n):
        x_pop[i] = np.array(pop[:,i::n])#奇数列表示X
	
    for i in range(n):
	    x[i] = x_pop[i].dot(2**np.arange(precisions)[::-1])/float(2**precisions-1)*(ranges[i][1]-ranges[i][0])+ranges[i][0]
    return x

def crossover_and_mutation(pop, CROSSOVER_RATE,POP_SIZE,precisions, MUTATION_RATE):
    """

    Returns
    -------
    new_pop : array
    对种群进行交叉与变异.

    """
    new_pop = []

    for father in pop:		#遍历种群中的每一个个体，将该个体作为父亲
        child = father		#孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:			#产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]	#再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=precisions*2)	#随机产生交叉的点
            child[cross_points:] = mother[cross_points:]		#孩子得到位于交叉点后的母亲的基因
        mutation(child, MUTATION_RATE,precisions)	#每个后代有一定的机率发生变异
        new_pop.append(child)
    return new_pop

def mutation(child, MUTATION_RATE,precisions):
    """

    变异

    """
    if np.random.rand() < MUTATION_RATE: 				#以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0,precisions)	#随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point]^1 	#将变异点的二进制为反转

def select(pop, fitness,POP_SIZE):  
    """

    Returns
    -------
    array
        重新选择种群.

    """
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness)/(fitness.sum()) )
    return pop[idx]

def get_fitness1(pop,F,n,precisions,ranges,POP_SIZE): 
    """

    Returns
    -------
    pred : array
        种群所对应函数值.

    """
    x = translateDNA(pop,n,ranges,precisions,POP_SIZE)
    pred = F(x)
    return pred 

def print_info(c,pop,F,n,precisions,ranges,POP_SIZE):
    """

    得到最优解

    """
    
    fitness = get_fitness1(pop,F,n,precisions,ranges,POP_SIZE)
    if c == 1:
	    best_fitness_index = np.argmax(fitness)
    if c == 0:
	    best_fitness_index = np.argmin(fitness)
    #print("optimal_value:", fitness[best_fitness_index])
    x = translateDNA(pop,n,ranges,precisions,POP_SIZE)
    return fitness[best_fitness_index],x[:,best_fitness_index:best_fitness_index+1]
    #for i in range(n):                  
	    #print(x[i][best_fitness_index])
    #print("最优的基因型：", pop[best_fitness_index])
	#print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))


def ga(c,F,n,ranges,precisions,N_GENERATIONS,POP_SIZE,MUTATION_RATE, CROSSOVER_RATE):

	pop = np.random.randint(n, size=(POP_SIZE, precisions*n)) #matrix (POP_SIZE, DNA_SIZE)
	for _ in range(N_GENERATIONS):#迭代N代
		#x = translateDNA(pop,F,n,precisions,ranges,POP_SIZE)
		pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE,POP_SIZE,precisions, MUTATION_RATE))
		#F_values = F(translateDNA(pop)[0], translateDNA(pop)[1])#x, y --> Z matrix
		fitness = get_fitness(c,pop,F,n,precisions,ranges,POP_SIZE)
		pop = select(pop, fitness,POP_SIZE) #选择生成新的种群
	
	return print_info(c,pop,F,n,precisions,ranges,POP_SIZE)
	#plt.ioff()
	#plot_3d(ax)
  
class GAModel(Model):
    def _init_(self):
        pass
    
    def fit(self,**kwargs):
        """
        Parameters
        ----------
        c:int,必选
        指定求函数的最大值或最小值，‘1’为求最大值，‘0’为求最小值
        F:function,必选
        指定所要求最值的函数表达式
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
        c = kwargs["c"]
        F = kwargs["F"]
        n = kwargs["n"]
        ranges = kwargs["ranges"]
        if "precisions" not in kwargs.keys():
            precisions = 24
        else :
            precisions = kwargs["precisions"]
        if "POP_SIZE" not  in kwargs.keys():
            POP_SIZE = 200
        else :
            POP_SIZE = kwargs["POP_SIZE"]
        if "MUTATION_RATE"  not in kwargs.keys():
            MUTATION_RATE = 0.005
        else :
            MUTATION_RATE = kwargs["MUTATION_RATE"]
        if "CROSSOVER_RATE" not  in kwargs.keys():
            CROSSOVER_RATE = 0.8
        else :
            CROSSOVER_RATE = kwargs["CROSSOVER_RATE"]
        if "N_GENERATIONS" not  in kwargs.keys():
            N_GENERATIONS = 50
        else :
            N_GENERATIONS = kwargs["N_GENERATIONS"]
        return ga(c,F,n,ranges,precisions,N_GENERATIONS,POP_SIZE,MUTATION_RATE, CROSSOVER_RATE)
    
    def predict(self, **kwargs):
        
        pass

if __name__ == '__main__': 
    
    def F(x):
	    return 3*(1-x[0])**2*np.exp(-(x[0]**2)-(x[1]+1)**2)- 10*(x[0]/5 - x[0]**3 - x[1]**5)*np.exp(-x[0]**2-x[1]**2)- 1/3**np.exp(-(x[0]+1)**2 - x[1]**2) - (x[2]-3)**2
    
    c = 1
    F = F
    n = 3
    ranges = np.array([[-3,3],[-3,3],[0,4]])
    #print(ranges.type)
    precisions = 24
    N_GENERATIONS = 50
    POP_SIZE = 200
    MUTATION_RATE = 0.005
    CROSSOVER_RATE = 0.8
    model = GAModel()
    model.fit(c=c,F=F,n=n,ranges=ranges,precisions=precisions,N_GENERATIONS=N_GENERATIONS,POP_SIZE=POP_SIZE,MUTATION_RATE=MUTATION_RATE, CROSSOVER_RATE=CROSSOVER_RATE)
    #ga(0,F,3,[[-3,3],[-3,3],[0,4]],24,50,200,0.005,0.8)
