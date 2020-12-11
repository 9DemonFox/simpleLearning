import numpy as np
import pandas as pd
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
    def __init__(self, _gamma, _lambda, max_depth):
        self._gamma = _gamma  # 正则化项中T前面的系数
        self._lambda = _lambda  # 正则化项w前面的系数
        # self.max_leaves = max_leaves
        self.root = None
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
        # print(lind)
        rind = list(set(range(X_data.shape[0])) - set(lind))
        # print(rind)
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

        if len(trees) > 0:
            sumval /= len(trees)
        else:
            return n

        for t in trees:
            for i in range(len(X_data)):
                tmplist.append(-garr[i] - t.predict(X_data[i]))  # X_data格式
        """
        for reslist in tmplist:
            for res in reslist:
                sumval += res**2
        """
        for res in tmplist:
            sumval += res ** 2

        currval = 0
        for i in range(len(X_data)):
            currval += (-garr[i] - self.predict(X_data[i])) ** 2

        tall = currval / (sumval / len(trees))
        nm = n * float(np.exp(-1)) ** tall
        # print("nm")
        # print(currval, sumval)
        return nm

    def calWeight(self, garr):
        # 计算叶节点权重，也即位于该节点上的样本预测值
        return - sum(garr) / self._lambda  # 修改过

    def calObj(self, garr):
        # 计算某个叶节点的目标(损失)函数值\
        # print("calobj")
        # print(garr,harr)
        # print((-1.0 / 2) * sum(garr) ** 2 / (sum(harr) + self._lambda) + self._gamma)
        return (-1.0 / 2) * sum(garr) ** 2 / self._lambda + self._gamma  # 修改过

    def getBestSplit(self, X_data, garr, splits):
        # 搜索最优切分点
        if not splits:
            return None
        else:
            bestSplit = None
            maxScore = -float('inf')
            score_pre = self.calObj(garr)  # 当前树叶节点的最佳损失值
            subinds = None
            for sp in splits:
                lind, rind = self.split(X_data, sp)
                if len(rind) < 2 or len(lind) < 2:
                    # print("short!")
                    continue
                gl = garr[lind]
                gr = garr[rind]
                # hl = harr[lind]
                # hr = harr[rind]
                # print(score_pre)
                score = score_pre - self.calObj(gl) - self.calObj(gr)  # 切分后目标函数值下降量
                # print(score, maxScore)
                if score > maxScore:
                    maxScore = score
                    bestSplit = sp  # 最佳切分点，元祖（feature, data）
                    subinds = (lind, rind)
            if maxScore < 0:  # pre-stopping
                return None
            else:
                return bestSplit, subinds

    def buildTree(self, X_data, garr, splits, depth):
        # 递归构建树
        res = self.getBestSplit(X_data, garr, splits)  # 最佳分割点
        depth += 1
        if not res or depth >= self.max_depth:
            return Node(w=self.calWeight(garr))
        bestSplit, subinds = res
        # print("bestSplit")
        # print(depth)
        splits.remove(bestSplit)
        # leaves += 1
        left = self.buildTree(X_data[subinds[0]], garr[subinds[0]], splits, depth + 1)  # splits减去本身节点，depth++
        right = self.buildTree(X_data[subinds[1]], garr[subinds[1]], splits, depth + 1)

        return Node(sp=bestSplit, right=right, left=left)

    def fit(self, X_data, garr):
        splits = self._candSplits(X_data)
        self.root = self.buildTree(X_data, garr, splits, 0)

    def predict(self, x):
        def helper(currentNode):
            if currentNode.isLeaf():
                return currentNode.w
            fea, val = currentNode.sp
            if x[fea] <= val:
                return helper(currentNode.left)
            else:
                return helper(currentNode.right)

        return helper(self.root)

    def _display(self):
        def helper(currentNode):
            if currentNode.isLeaf():
                print(currentNode.w)
            else:
                print(currentNode.sp)
            if currentNode.left:
                helper(currentNode.left)
            if currentNode.right:
                helper(currentNode.right)

        helper(self.root)


class IBRTModel(Model):
    def __init__(self, n_iter, _gamma, _lambda, max_depth):
        self.n_iter = n_iter  # 迭代次数，即基本树的个数
        self._gamma = _gamma
        self._lambda = _lambda
        # self.max_depth = max_depth #最大端节点数
        self.max_depth = max_depth  # 单颗基本树最大深度
        # self.eta = 1.0#[]  # 收缩系数, 默认1.0,即不收缩
        self.trees = []
        self.eta_trees = []

    def calGrad(self, y_pred, y_data):
        # 计算一阶导数
        return np.sign(y_data - y_pred)

    # def calHess(self, y_pred, y_data):
    #    return 2 * np.ones_like(y_data)

    # def calEtaTree(self, trees):

    def fit(self, X_data, y_data):
        step = 0
        while step < self.n_iter:
            # if step==0:

            tree = Tree(self._gamma, self._lambda, self.max_depth)
            y_pred = self.predict(X_data)  # 会调用Tree.predict()（前step-1轮的）

            garr = self.calGrad(y_pred, y_data)

            tree.fit(X_data, garr)

            self.eta_trees.append(tree.calEtaTree(self.trees, garr, X_data, 0.005))

            self.trees.append(tree)

            step += 1

    def predict(self, X_data):
        if self.trees:
            y_pred = []
            # print("et")
            # print(self.eta_trees)
            for x in X_data:
                y_pred.append(sum([self.eta_trees[i] * self.trees[i].predict(x) for i in range(len(self.trees))]))
            return np.array(y_pred)
        else:
            return np.zeros(X_data.shape[0])


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import matplotlib.pyplot as plt

    ibrt = IBRTModel(10, 0, 1.0, 2)
    ibrt_loader = IBRTDataLoader(datapath="../data/ibrt/test.xlsx")

    trainX, trainY = ibrt_loader.loadTrainData()
    testX, testY = ibrt_loader.loadTestData()

    ibrt.fit(trainX, trainY)
    predictY = ibrt.predict(testX)
    print(mean_absolute_error(testY, predictY))




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



