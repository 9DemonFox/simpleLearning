import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from data.ibrt.dataLoader import IBRTDataLoader
from model import Model
""""
class IBRTModel(Model):
    def __init__(self, **kwargs):
        
        #self.model = GradientBoostingRegressor(**kwargs)

    def fit(self, **kwargs):
        assert "trainX" in kwargs.keys()
        assert "trainY" in kwargs.keys()
        trainX = kwargs["trainX"]
        trainY = kwargs["trainY"]
        self.model.fit(trainX, trainY)

    def predict(self, **kwargs):
        assert "predictX" in kwargs.keys()
        predictX = kwargs["predictX"]
        return self.model.predict(predictX)
"""
def fra_Data(data, fra):  # fra为正整数
    ret = data.copy()
    while fra > 1:
        ret = np.concatenate((ret, data), axis=0)
        fra -= 1
    return ret


class Node:
    def __init__(self, sp=None, left=None, right=None, w=None):
        self.sp = sp  # 非叶节点的切分，特征以及对应的特征下的值组成的元组
        self.left = left
        self.right = right
        self.w = w  # 叶节点权重，也即叶节点输出值

    def isLeaf(self):
        return self.w


class Tree:
    def __init__(self, max_depth):
        # self._gamma = _gamma  # 正则化项中T前面的系数
        # self._lambda = _lambda  # 正则化项w前面的系数
        # self.max_leaves = max_leaves
        self.max_depth = max_depth
        self.root = None
        self.numofleaves = 0
        # self.m_eta = 1.0

    def _candSplits(self, X_data):
        # 计算候选切分点
        splits = []
        for fea in range(X_data.shape[1]):
            for val in X_data[:, fea]:
                splits.append((fea, val))
        # print(splits)
        return splits

    def split(self, X_data, sp):
        # 劈裂数据集，返回左右子数据集索引（行索引）
        lind = np.where(X_data[:, sp[0]] <= sp[1])[0]
        rind = list(set(range(X_data.shape[0])) - set(lind))
        return lind, rind

    """def calEta(self, garr, X_data):
                    #t = self.
                    eta_trees = []
                    for t in trees:
                        eta_trees.append(sum((-garr-t.predict(X_data))**2))
                    t = sum((-garr-self.predict(X_data))**2)/"""

    def calEtaTree(self, trees, garr, X_data, n):
        tmplist = []
        # ret_etatrees = []
        sumval = 0
        # print(garr)
        if len(trees) > 0:
            sumval /= len(trees)
        else:
            return n

        for t in range(len(trees)):
            for i in range(len(X_data)):
                #print('garr:',garr)
                #print('t,len(trees):',t,len(trees))
                tmplist.append(-garr[t][i] - trees[t].predict(X_data[i]))  # X_data格式
        """
        for reslist in tmplist:
            for res in reslist:
                sumval += res**2
        """
        for res in tmplist:
            sumval += res ** 2

        currval = 0  # 分子
        for i in range(len(X_data)):
            currval += (-garr[-1][i] - self.predict(X_data[i])) ** 2

        tall = currval / (sumval / len(trees))
        nm = n * float(np.exp(-1)) ** tall
        # print("nm")
        # print('nm:',nm)
        # print('currval, sumval:',currval, sumval)
        return nm

    def calWeight(self, garr, y_data, y_pred):
        # 计算叶节点权重，也即位于该节点上的样本预测值
        ebcl = 0.0
        # 排序y_data-y_pred
        l = (y_data - y_pred)

        l.sort()
        # print('l:',l)
        # fun = lambda x : sum(np.abs(y_data-y_pred-x)-ebcl) #ebcl=0.0
        # res = minimize(fun, 0, method='SLSQP')
        # print('weight:', l[int(len(l)/2)])
        return l[int(len(l) / 2)]
        # return - sum(garr) / self._lambda #修改过

    # 需仔细看这个函数！！！
    def calObj(self, garr, y_data, y_pred):
        # 计算某个叶节点的目标(损失)函数值\
        # print("calobj")
        # print(garr,harr)
        # print((-1.0 / 2) * sum(garr) ** 2 / (sum(harr) + self._lambda) + self._gamma)
        # print(garr)
        # print('sizeofgarr:',len(garr))
        # print('y_data:',y_data)
        # print('y_pred:',y_pred)
        ebcl = 0.0
        # fun = lambda x : sum(np.abs(y_data-y_pred-x)-ebcl) #ebcl
        # res = minimize(fun, 0, method='SLSQP')
        l = (y_data - y_pred)
        l.sort()
        res_x = l[int(len(l) / 2)]

        # print('success?:',res.success)
        # print('res.fun:',res.fun)
        return sum(np.abs(y_data - y_pred - res_x) - ebcl)
        # return (-1.0 / 2) * sum(garr) ** 2 / self._lambda + self._gamma * self.numofleaves #修改过

    # 需仔细看这个函数！！！
    def getBestSplit(self, X_data, garr, splits, y_data, y_pred):
        # 搜索最优切分点
        if not splits:
            return None
        else:
            bestSplit = None
            maxScore = -float('inf')
            score_pre = self.calObj(garr, y_data, y_pred)  # 当前树叶节点的最佳损失值
            # print('sc_pre:%f'%score_pre)
            subinds = None
            for sp in splits:
                # print('sp:',sp)
                lind, rind = self.split(X_data, sp)

                if len(rind) < 2 or len(lind) < 2:
                    # print("short!")
                    continue


                gl = garr[lind]
                gr = garr[rind]
                # hl = harr[lind]
                # hr = harr[rind]
                # print('garr:',garr)
                # print('sc:bef:',score_pre)
                # print('sc_aft:%f'%(self.calObj(gl) + self.calObj(gr)))
                # print('getsp!!!!!!')

                score = score_pre - self.calObj(gl, y_data, y_pred) - self.calObj(gr, y_data, y_pred)  # 切分后目标函数值下降量
                # print('score, maxScore:',score, maxScore)
                if score > maxScore:
                    maxScore = score
                    bestSplit = sp  # 最佳切分点，元祖（feature, data）
                    subinds = (lind, rind)
            #print('bestSplit:', bestSplit)
            #print('maxscore:%f' % maxScore)
            if maxScore < 0:  # pre-stopping
                return None
            else:
                return bestSplit, subinds

    def buildTree(self, X_data, garr, splits, depth, y_data, y_pred):
        # 递归构建树
        res = self.getBestSplit(X_data, garr, splits, y_data, y_pred)  # 最佳分割点
        depth += 1
        # print("depth,res:",depth,res)
        # print(res)#res都为None
        if not res or depth >= self.max_depth:
            # print("w:",self.calWeight(garr))
            self.numofleaves += 1
            return Node(w=self.calWeight(garr, y_data, y_pred) + 0.0001)  # 为了防止把w结果0视为None

        # print("bestSplit:",res[0])
        bestSplit, subinds = res
        # print("bestSplit")
        # print(depth)
        splits.remove(bestSplit)
        # leaves += 1
        left = self.buildTree(X_data[subinds[0]], garr[subinds[0]], splits, depth, y_data,
                              y_pred)  # splits减去本身节点，depth++
        right = self.buildTree(X_data[subinds[1]], garr[subinds[1]], splits, depth, y_data, y_pred)

        return Node(sp=bestSplit, right=right, left=left)

    def fit(self, X_data, garr, y_data, y_pred):
        splits = self._candSplits(X_data)
        self.root = self.buildTree(X_data, garr, splits, 0, y_data, y_pred)

    def predict(self, x):  # 这里x为一维
        def helper(currentNode):
            if currentNode.isLeaf():
                return currentNode.w  # 目前问题：w太小，导致tree.predict()太小e-6级别

            # print('currentNode.sp:',currentNode.sp)
            # print('_display:',self._display())
            fea, val = currentNode.sp
            if x[fea] <= val:
                return helper(currentNode.left)
            else:
                return helper(currentNode.right)

        return helper(self.root)

    def _display(self):
        def helper(currentNode):
            if currentNode.isLeaf():
                print('currentNode.w:', currentNode.w)
            else:
                print('currentNode.sp:', currentNode.sp)
            if currentNode.left:
                helper(currentNode.left)
            if currentNode.right:
                helper(currentNode.right)

        helper(self.root)


class IBRTModel(Model):
    # 去掉了_gamma,_lambda正则化参数
    def __init__(self, n_iter, max_depth):
        self.n_iter = n_iter  # 迭代次数，即基本树的个数
        # self._gamma = _gamma
        # self._lambda = _lambda
        # self.max_depth = max_depth #最大端节点数
        self.max_depth = max_depth  # 单颗基本树最大深度
        # self.eta = 1.0#[]  # 收缩系数, 默认1.0,即不收缩
        self.trees = []
        self.eta_trees = []
        self.model = sklearn.ensemble.GradientBoostingRegressor(max_depth=max_depth, n_estimators=n_iter, loss='huber', max_features=3, max_leaf_nodes=3, subsample=0.8)


    def calGrad(self, y_pred, y_data):
        # 计算一阶导数，ebcl不敏感损失函数
        len_y = len(y_data)
        ebcl = 3 * np.std(y_data - y_pred, ddof=1) * np.sqrt(np.log(len_y) / len_y)  # 为了继续/10
        #print('ebcl：', ebcl)
        ret = []
        for i in y_data - y_pred:
            # print('y_data',y_data)
            # print('y_pred',y_pred)
            # print('i,ebcl',i,ebcl)
            if np.abs(i) >= ebcl:
                ret.append(np.sign(i))
            else:
                ret.append(0)
        # print(np.array(ret))
        return np.array(ret)


    def fit(self, **kwargs):

        trainX = kwargs["trainX"]
        trainY = kwargs["trainY"]

        step = 0
        garr = []

        self.model.fit(trainX,trainY)

        '''
        while step < self.n_iter:
            tree = Tree(self.max_depth)
            if step == 0:
                #fun = lambda y: sum(np.abs(trainY - y))  # ebcl
                l = trainY.copy()
                l.sort()
                y_pred = np.full(len(l), l[int(len(l) / 2)])
            else:
                y_pred += self.predict(predictX=trainX) # 会调用Tree.predict()（前step-1轮的）
            # print('残差:%f' % (np.mean(y_data - y_pred)))
            garr.append(self.calGrad(y_pred, trainY))
            tree.fit(trainX, garr[-1], trainY, y_pred)  # 每次迭代由于garr不一样，迭代结果不一样
            tree._display()
            self.eta_trees.append(tree.calEtaTree(self.trees, garr, trainX, 0.01))

            self.trees.append(tree)
            
            
            for t in self.trees:
                print('t.sp')
                print(t.root.sp)
                print('tree_pre:%f'%t.predict(X_data[0]))
            '''
            #step += 1

    def predict(self, **kwargs):
        assert "predictX" in kwargs
        X_data = kwargs.get("predictX")
        #print('pred:', X_data)
        return self.model.predict(X_data)

        '''
        if self.trees:
            y_pred = []
            # print("et")
            # print(self.eta_trees)
            for x in X_data:
                y_pred.append(sum(self.eta_trees[i] * self.trees[i].predict(x) for i in range(len(self.trees))))
                # print('y_pred:', y_pred)
            return np.array(y_pred).flatten()
        else:
            return np.zeros(X_data.shape[0])
        '''
    def fitForUI(self, **kwargs):
        """ 返回结果到前端
        :return:
        """
        self.fit(**kwargs)
        # 返回结果为字典形式
        #excludeFeatures, coefs = self.fit(*args)
        returnDic = {
            "特征重要性": str(self.model.feature_importances_),

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
        predictResult = self.predict(predictX=testX)
        mse = mean_squared_error(predictResult, testY)
        mae = mean_absolute_error(predictResult, testY)
        returnDic["预测结果"] = str(predictResult)
        returnDic["mean_absolute_error"] = str(mae)
        returnDic["mean_squared_error"] = str(mse)
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



