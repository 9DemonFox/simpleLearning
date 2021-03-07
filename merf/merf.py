
import pandas as pd 
import numpy as np
from data.merf.dataLoder import MERFDataLoader
from model import Model
from merf.mf import MERF
from sklearn.metrics import mean_squared_error



class MERFModel(Model):

    def __init__(self, **kwargs):
        """
        
        n:int,可选(默认 = 1)
        观测对象种类数量
        q：int
        随机效应变量数
        D:array
        随机效应变量的协方差矩阵
        u:array
        随机效应变量
        σ2:float
        误差的方差 
        epoch:int,可选(默认=50)
        循环轮数
        k:int,可选(默认=1)
        表示第k个变量作为随机效应变量

        """
        if "n" not in kwargs.keys():
            self.n = 1
        else:
            self.n = kwargs["n"]

        if "k" not in kwargs.keys():
            self.k = 1
        else:
            self.k = kwargs["k"]
        if "ga" not in kwargs.keys():
            self.ga = 0
        else:
            self.ga = kwargs["ga"]

    def fitForUI(self, **kwargs):
        """ 返回结果到前端
        :return:
        """
        # 返回结果为字典形式
        if (self.ga==0):  
            b,e = self.fit(**kwargs)
            returnDic = {
            "随机效应系数": str(np.array(b)),
            "误差": str(e),
        }
        if (self.ga==1):  
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
              "k": str(x),
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
        if(self.ga!=1):
            self.merf = MERF()
            N = self.trainX.shape[0]
            m = int(N / self.n)
            z = np.ones((N,1))
            z = z + self.trainX[:, self.k-1:self.k]
            a = np.ones((m)).astype(int)
            c = pd.Series(a)
            for i in range(self.n):
                if i!=0:
                    a = (i+1)*np.ones((m)).astype(int)
                    d = pd.Series(a)
                    c = pd.concat([c,d], ignore_index=True)
            _,b,e = self.merf.fit(self.trainX, z, c, self.trainY)
            return b,e
        
       

    def test(self, **kwargs):
        self.testX = kwargs["testX"]
        self.testY = kwargs["testY"]
        if (self.ga==1):
            score = np.zeros((self.trainX.shape[1]))
            #print(self.trainX.shape)
            for i in range(self.trainX.shape[1]):
                model1 = MERFModel(n=self.n, k=i+1, ga=0)
                model1.fit(trainX=self.trainX, trainY=self.trainY)
                score[i] = model1.test(testX=self.testX, testY=self.testY)     
            self.k = np.argmax(score) + 1  
            self.model = MERFModel(n=self.n, k=self.k)
            self.model.fit(trainX=self.trainX, trainY=self.trainY)
            N = self.testX.shape[0]
            x0 = self.model.predict(predictX=self.testX).reshape(N, 1)
            mse = mean_squared_error(x0, self.testY)
            x = self.k
            return x,mse
        N = self.testX.shape[0]
        m = int(N / self.n)
        z = np.ones((N,1))
        z = z + self.testX[:, self.k-1:self.k]
        a = np.ones((m)).astype(int)
        c = pd.Series(a)
        for i in range(self.n):
            if i!=0:
                a = (i+1)*np.ones((m)).astype(int)
                d = pd.Series(a)
                c = pd.concat([c,d], ignore_index=True)
        x0 = self.merf.predict(self.testX, z, c).reshape(N, 1)  
        return mean_squared_error(x0, self.testY)
    
    def predict(self, **kwargs):
        self.predictX = kwargs["predictX"]
        if self.ga==1:
            self.merf = MERF()
            N = self.trainX.shape[0]
            m = int(N / self.n)
            z = np.ones((N,1))
            z = z + self.trainX[:, self.k-1:self.k]
            a = np.ones((m)).astype(int)
            c = pd.Series(a)
            for i in range(self.n):
                if i!=0:
                    a = (i+1)*np.ones((m)).astype(int)
                    d = pd.Series(a)
                    c = pd.concat([c,d], ignore_index=True)
            self.merf.fit(self.trainX, z, c, self.trainY)
        N = self.predictX.shape[0]
        m = int(N / self.n)
        z = np.ones((N,1))
        z = z + self.predictX[:, self.k-1:self.k]
        a = np.ones((m)).astype(int)
        c = pd.Series(a)
        for i in range(self.n):
            if i!=0:
                a = (i+1)*np.ones((m)).astype(int)
                d = pd.Series(a)
                c = pd.concat([c,d], ignore_index=True)
        x0 = self.merf.predict(self.predictX, z, c).reshape(N, 1)
        return x0


if __name__ == '__main__':
    n = 1
    k = 1
    datapath1="../data/merf/data_train1.xlsx"
    datapath2="../data/merf/data_test1.xlsx"
    datapath3="../data/merf/data_predict1.xlsx"
    dataloader = MERFDataLoader()
    trainX, trainY = dataloader.loadTrainData(train_path=datapath1)
    testX, testY = dataloader.loadTestData(test_path=datapath2)
    predictX = dataloader.loadPredictData(predict_path=datapath3)  
    model = MERFModel(n=n, k=k ,ga=0)
    print(model.fitForUI(trainX=trainX, trainY=trainY))
    print(model.testForUI(testX=testX, testY=testY))
    print(model.predictForUI(predictX=predictX))
    