import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import random


from data.ibrt.dataLoader import IBRTDataLoader
from model import Model

def fra_Data(data, fra):  # fra为正整数
    ret = data.copy()
    while fra > 1:
        ret = np.concatenate((ret, data), axis=0)
        fra -= 1
    return ret

def findBestFeatureAndPoint(node, ebcl):
    '''
    依据MSE准则，找到最佳切分特征和最佳切分点
    :param node: 进行分裂的节点, 一个矩阵
    :return: 切分特征与切分点
    '''

    # n为特征数
    m,n=node.shape
    # 因为最后一列是标签值
    n=n-1
    # 需要预测的真实值
    y = node[:, -1]

    # 用来保存最佳切分特征与切分点
    # 以及左右子树
    min_loss = np.Inf
    best_feature = -1
    best_point = -1
    best_left=None
    best_right=None



    # 找到最佳切分特征与切分点
    # 我们遍历所有特征，然后遍历该特征所有（或者部分）切分点
    # 取决于该特征是离散还是连续变量
    for feature in range(n):
        # 注意是n-1 ， 因为最后一个是样本需要预测的值

        # 获得进行切分列
        # 因为是连续数据，有可能有很多不同的值
        # 所以此处我们进行切分的时候，若是离散数据（默认种类小于等于10），我们进行精确切分
        # 若类型大于10，认为是连续变量，进行10分位点切分
        column=node[:,feature]
        category=sorted(set(column))
        if len(category)<=10:
            split_point=category
        else:
            # 使用np.arrange来每次找到1/10数据点所在的索引
            # 然后进行切分
            split_point = np.arange(0, len(category), len(category) // 10)
            split_point = [category[split_point[i]] for i in range(0, len(split_point))]



        # 确定了所有切分点之后，对切分点进行遍历，找到最佳切分点
        for point in split_point:
            # 尝试切分
            left=column[column<=point]
            right=column[column>point]

            # 左右两边的需要预测的真实值
            y_left=y[column<=point]
            y_right=y[column>point]


            # 带参数ebcl的损失函数
            c_left = np.average(y_left)
            c_right = np.average(y_right)
            ll = y_left - c_left
            for l in np.abs(ll):
                if l<ebcl:
                    l = 0

            rr = y_right - c_right
            for r in np.abs(rr):
                if r < ebcl:
                    l = 0


            #loss=np.sum(np.square(y_left-c_left))+np.sum(np.square(y_right-c_right))
            loss = np.sum(np.abs(y_left-c_left))+np.sum(np.abs(y_right-c_right))

            if loss<min_loss:
                min_loss=loss
                best_feature=feature
                best_point=point
                best_left=node[column<=point]
                best_right=node[column>point]
    return (best_feature,best_point,best_left,best_right)







def createCART(data,deep,max_deep=2, ebcl=0.2):
    '''
    创建回归树，分裂准则MSE（最小均方误差）
    :param deep: 树的当前深度
    :param max_deep:  树的最大深度（从0开始），默认为2，即产生4个叶子节点
    :param data: 训练样本，其中data中的最后一列值为上一轮训练之后的残差
    :return: 一颗回归树
    '''

    # 树的结构例如
    # tree={3:{'left':{4:{'left':23.1,'right':19.6},'point':0},'right':{6:{'left':23.1,'right':19.6},'point':4.5}},'point':10.4}
    # 上面是一颗2层的回归树
    # 3代表根节点以第三个特征进行分类，分裂的切分点是point=10.4
    # 然后是左右子树left，right
    # left也是一个字典，对应左子树
    # 4代表左子树以特征四为分裂特征，切分点是point=0
    # 分裂之后的left仍然是一个字典，其中有left和right对应着23.1,19.6
    # 这两个值即为我们的预测值
    # 右子树也同理

    if deep<=max_deep:
        feature,point,left,right=findBestFeatureAndPoint(data, ebcl)
        tree = {feature: {}}
        if deep!=max_deep:
            # 不是最后一层，继续生成树
            tree['point']=point
            if len(left)>=2:
                # 必须要保证样本长度大于1，才能分裂
                tree[feature]['left']=createCART(left,deep+1,max_deep,ebcl)
            else:
                tree[feature]['left']=np.average(left)
            if len(right)>=2:
                tree[feature]['right']=createCART(right,deep+1,max_deep,ebcl)
            else:
                tree[feature]['right']=np.average(right)

        else:
            # feature, point, left, right = findBestFeatureAndPoint(data)
            # tree['point']=point
            # # y标签在训练样本最后一列，用-1获取
            # y_left=left[:,-1]
            # y_right=right[:,-1]
            # c_left = np.average(y_left)
            # c_right = np.average(y_right)

            # 最后一层树，保存叶节点的值
            return np.average(data[:,-1])
        return tree


def gradientBoosting(round, data0, alpha, fra, ebcl):
    '''

    :param round: 迭代论数，也就是树的个数
    :param data: 训练集
    :param alpha: 防止过拟合，每一棵树的正则化系数
    :return:
    '''
    #print('data:',data.shape)
    # 扩充样本，7倍
    data = fra_Data(data0, 7)
    tree_list=[]
    # 第一步，初始化fx0，即找到使得损失函数最小的c
    # -1 代表没有切分特征，所有值均预测为样本点均值
    fx0={-1:np.average(data0[:,-1])}

    tree_list.append(fx0)
    # 开始迭代训练，对每一轮的残差拟合回归树
    import time
    from UI import Controler

    for i in range(1,round):

        time.sleep(0.5)  # 增加训练时间，显示进度条效果
        Controler.PROGRESS_NOW = int((95 / round) * i)
        print(Controler.PROGRESS_NOW)
        # 更新样本值，rmi=yi-fmx


        print('Step ', i)
        #print(data0.shape[0],)
        # 每轮随机进行采样
        choice_ind = random.sample(range(0,data.shape[0]), int(fra * data0.shape[0]))
        data = data[choice_ind,:]
        if i==1:
            data[:,-1]=data[:,-1]-fx0[-1]
        else:
            for i in range(len(data)):
                # 注意，这里穿的列表是tree_list中最后一个
                # 因为我们只需要对残差进行拟合，data[:,-1]每一轮都进行了更新，所以我们只要减去上一颗提升树的预测结果就是残差了
                data[i, -1] = data[i, -1] - predict_for_rm(data[i], tree_list[-1], alpha)
        # 上面已经将样本值变为了残差，下面对残差拟合一颗回归树
        fx = createCART(data, deep=0, max_deep=4, ebcl=ebcl)
        #
        # 将树添加到列表
        tree_list.append(fx)
    #print('tl:',tree_list)
    return tree_list


def predict_for_rm(data, tree, alpha):
    '''
    获得前一轮 第m-1颗树 的预测值，从而获得残差
    :param data: 一条样本
    :param tree: 第 m-1 颗树
    :param alpha: 正则化系数
    :return:  第m-1颗树预测的值
    '''

    while True:
        # 遍历该棵树，直到叶节点
        # 叶节点与子树的区别在于一节点上的值为float
        # 而子树是一个字典，有point键，用作切分点
        # tree={3:{'left':{4:{'left':23.1,'right':19.6},'point':0},'right':{6:{'left':23.1,'right':19.6},'point':4.5}},'point':10.4}
        #
        if type(tree).__name__=='dict':
            # 如果是字典，那么这是一颗子树,
            point = tree['point']
            # tree.keys()=dict_keys([3, 'point'])
            # 所以int值对应的是特征，但是字典的键值是无序的，我们无法保证第一个是特征，所以用类型来判断
            feature = list(tree.keys())[0] if type(list(tree.keys())[0]).__name__ == 'int' else list(tree.keys())[1]
            if data[feature] <= point:
                tree = tree[feature]['left']
            else:
                tree = tree[feature]['right']
        else:
            # 当tree中没有切分点point，证明这是一个叶节点，tree就是预测值，返回获得预测值
            #print('pre:', alpha * tree)
            return alpha * tree




def pre(data, tree_list, alpha):
    '''
    对一条样本进行预测
    :param tree_list: 所有树的列表
    :param data: 一条需要预测的样本点
    :param alpha:正则化系数
    :return: 预测值
    '''
    #print("输入Xdata:", data)
    #print('输入treelist:', tree_list)
    #print('alpha:', alpha)
    m=len(tree_list)
    fmx=0
    for i in range(m):
        tree=tree_list[i]
        #print('tree',tree)
        if i==0:
            #  fx0={-1:np.average(data[:,-1])}
            # fx0是一个叶节点，只有一个预测值，树的深度为0
            fmx+=tree[-1]
        else:
            while True:
                # 遍历该棵树，直到叶节点
                # 叶节点与子树的区别在于一节点上的值为float
                # 而子树是一个字典，有point键，用作切分点
                # tree={3:{'left':{4:{'left':23.1,'right':19.6},'point':0},'right':{6:{'left':23.1,'right':19.6},'point':4.5}},'point':10.4}
                #
                if type(tree).__name__=='dict':
                    # 如果是字典，那么这是一颗子树,
                    point=tree['point']
                    # tree.keys()=dict_keys([3, 'point'])
                    # 所以int值对应的是特征，但是字典的键值是无序的，我们无法保证第一个是特征，所以用类型来判断
                    feature=list(tree.keys())[0] if type(list(tree.keys())[0]).__name__=='int' else list(tree.keys())[1]
                    if data[feature]<=point:
                        tree=tree[feature]['left']
                    else:
                        tree=tree[feature]['right']
                else:
                    # 当tree中没有切分点point，证明这是一个叶节点，tree就是预测值，返回获得预测值
                    fmx+= alpha * tree
                    break
    return fmx


def test(X_test, y_test, tree_list, alpha):
    acc = 0  # 正确率
    acc_num = 0  # 正确个数
    y_predict=[]
    for i in range(len(X_test)):
        print('testing ***', i)
        x = X_test[i]
        y_pred =pre(x, tree_list, alpha)
        y_predict.append(y_pred)
        if y_pred/y_test[i]<1.25 and y_pred/y_test[i]>0.8:
            acc_num += 1
        print(f'testing {i}th data :y_pred={y_pred},y={y_test[i]}')
        print('now_acc=', acc_num / (i + 1))
    return y_predict


class IBRTModel(Model):
    # 去掉了_gamma,_lambda正则化参数
    def __init__(self, n_iter, max_depth, alpha, fra, ebcl):
        self.n_iter = n_iter  # 迭代次数，即基本树的个数
        self.fra = fra  # 随机抽样系数
        self.max_depth = max_depth  # 单颗基本树最大深度
        # self.eta = 1.0#[]  # 收缩系数, 默认1.0,即不收缩
        #self.trees = []
        #self.eta_trees = []
        self.alpha = alpha
        self.ebcl = ebcl



    def fit(self, **kwargs):

        trainX = kwargs["trainX"]
        trainY = kwargs["trainY"]

        datay = trainY.reshape((-1, 1))
        data = np.concatenate((trainX, datay), axis=1)

        self.tree_list = gradientBoosting(self.n_iter, data, self.alpha, self.fra, self.ebcl)
        from UI import Controler
        Controler.PROGRESS_NOW = 100



    def predict(self, **kwargs):
        assert "predictX" in kwargs
        X_data = kwargs.get("predictX")

        #return self.model.predict(X_data)
        pred_y = []
        for onedata in X_data:
            #print('onedata',onedata)
            #print('oneprd',self.predone(onedata))
            pred_y.append(pre(onedata, self.tree_list, self.alpha))
            #print('pred_y:', pred_y)
        return pred_y

    def fitForUI(self, **kwargs):
        """ 返回结果到前端
        :return:
        """
        self.fit(**kwargs)
        # 返回结果为字典形式
        #excludeFeatures, coefs = self.fit(*args)
        returnDic = {
            "树(示例)": str(self.tree_list[0]),

        }
        return returnDic

    def testForUI(self, **kwargs):
        """
        :param args:
        :return: 字典形式结果
        """
        testX = kwargs["testX"]
        testY = kwargs["testY"]

        returnDic = {
            "mean_squared_error": None,
            "mean_absolute_error": None
        }
        #args["predictX"] = args[0]

        acc = 0  # 正确率
        acc_num = 0  # 正确个数
        y_predict = []
        for i in range(len(testX)):
            print('testing ***', i)
            x = testX[i]
            y_pred = pre(x)
            y_predict.append(y_pred)
            if y_pred / testX[i] < 1.25 and y_pred / testY[i] > 0.8:
                acc_num += 1
            print(f'testing {i}th data :y_pred={y_pred},y={testY[i]}')
            print('now_acc=', acc_num / (i + 1))
        mse = mean_squared_error(y_predict, testY)
        mae = mean_absolute_error(y_predict, testY)
        returnDic["预测结果"] = str(y_predict)
        returnDic["mean_absolute_error"] = str(mae)
        returnDic["mean_squared_error"] = str(mse)
        return returnDic

        '''
        predictResult = self.predict(predictX=testX)
        mse = mean_squared_error(predictResult, testY)
        mae = mean_absolute_error(predictResult, testY)
        
        returnDic["预测结果"] = str(predictResult)
        returnDic["mean_absolute_error"] = str(mae)
        returnDic["mean_squared_error"] = str(mse)
        '''
        return returnDic

    def predictForUI(self, **kwargs):
        """
        :param args:
        :return: 字典形式结果
        """
        returnDic = {
            "预测结果": None
        }
        predictResult = self.predict(**kwargs)
        returnDic["预测结果"] = str(predictResult)
        return returnDic

if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import matplotlib.pyplot as plt

    model = IBRTModel(5, 2)
    ibrt_loader = IBRTDataLoader()

    trainX, trainY = ibrt_loader.loadTrainData(train_path="../data/ibrt/IBRT_TRAIN_DATA.xlsx")
    testX, testY = ibrt_loader.loadTestData(test_path="../data/ibrt/IBRT_TEST_DATA.xlsx")
    print(trainX.shape,testX.shape)

    #model.fit(trainX=trainX, trainY=trainY)

    #print(mean_absolute_error(testY, predictY))

    model.fitForUI(trainX=trainX, trainY=trainY)
    predictY = model.predict(predictX=testX)
    predictYForUI = model.predictForUI(predictX=testX)
    predictTrainY = model.predict(predictX=trainX)
    #print(model.model.coef_)
    print('mean_squared_error of predictY, testY:',mean_squared_error(predictY, testY))
    print('mean_squared_error of predictTrainY, trainY:',mean_squared_error(predictTrainY, trainY))



    """data = pd.read_excel("../data/ibrt/test.xlsx")
    valid_data = np.array(data.iloc[1:])
    valid_data = fra_Data(valid_data, 3)
    X = valid_data[:, 1:-1]
    y = valid_data[:, -1]
    print(X)"""

    #10折交叉验证
    """mean_absolute_error_list = []
    kf = KFold(n_splits=10, shuffle=False)
    for train_index, test_index in kf.split(X):
        print('train_index:%s , test_index: %s ' % (train_index, test_index))
        trainX = X[train_index]
        testX = X[test_index]
        trainY = y[train_index]
        testY = y[test_index]
        f = IBRTModel(100, 0, 1.0, 2)
        f.fit(trainX, trainY)
        predictY = f.predict(testX)
        print(predictY)
        mean_absolute_error_list.append(mean_absolute_error(testY, predictY))
    print(np.mean(mean_absolute_error_list))"""



