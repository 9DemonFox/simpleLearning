import tkinter as tk
from tkinter import filedialog
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from mert.mert import MERTModel
from rebet.rebet import REBETModel
from ibrt.ibrt import IBRTModel
from gbm.GBM import GBMModel
from salp.SALP import SVRModel

from data.mert.dataLoder import MERTDataLoder
from data.rebet.dataLoder import REBETDataLoder
from data.ibrt.dataLoader import IBRTDataLoader
from data.gbm.dataLoader import GBMDataLoader
from data.salp.dataLoder import SalpDataLoder

class simplelearning:
    #界面布局方法
    def __init__(self):
        self.initWindow()
        #创建主界面，并且保存到成员属性中
        self.inibutton()
    
    def run(self):
        self.win.mainloop()
    
    def initWindow(self):
        self.win = tk.Tk()
        self.win.title("Python GUI")  # 添加标题
        self.win.geometry("{}x{}+{}+{}".format(1024, 768, 100, 0))
        
    def trainfile(self,filepath1):
        self.trainfile = filedialog.askopenfilename()
        filepath1.insert(0,self.trainfile)
        
    def testfile(self,filepath2):
        self.testfile = filedialog.askopenfilename()
        filepath2.insert(0,self.testfile)

    def inibutton(self):
        A = tk.Button(self.win, text ="MERT", command = self.createMERT)
        A.pack()
        B = tk.Button(self.win, text ="REBET", command = self.createREBET)
        B.pack()
        C = tk.Button(self.win, text ="IBRT", command = self.createIBRT)
        C.pack()
        D = tk.Button(self.win, text ="GBM", command = self.createGBM)
        D.pack()
        E = tk.Button(self.win, text ="Salp", command = self.createSalp)
        E.pack()
        
    def quittop(self,top):
        self.win.deiconify()
        top.destroy()
        del self.trainfile 
        del self.testfile 
    
    def createMERT(self):
        self.win.withdraw()
        top = tk.Toplevel()
        top.title('MERT')
        top.geometry("{}x{}+{}+{}".format(1024, 768, 100, 0))
        filepath1 = tk.Entry(top)
        filepath1.pack()
        file1 = tk.Button(top, text ="训练数据", command=lambda :self.trainfile(filepath1))
        file1.pack() 
        filepath2 = tk.Entry(top)
        filepath2.pack()
        file2 = tk.Button(top, text ="测试数据", command=lambda :self.testfile(filepath2))
        file2.pack() 
        n1 = tk.Label(top, text="类别数(默认值为1)")
        n1.pack()
        n2 = tk.Entry(top)
        n2.pack() 
        epoch1 = tk.Label(top, text="迭代次数(默认值为100)")
        epoch1.pack()
        epoch2 = tk.Entry(top)
        epoch2.pack() 
        k1 = tk.Label(top, text="选择第几个变量与随机效应相关(默认值为1)")
        k1.pack()
        k2 = tk.Entry(top)
        k2.pack() 
        n = n2.get()
        epoch = epoch2.get()
        k = k2.get()
        start = tk.Button(top, text ="开始", comman=lambda :self.startMERT(n,epoch,k,EditText))
        start.pack() 
        EditText = tk.Text(top,width=20,height=10)
        EditText.place(x=512,y=380)
        end = tk.Button(top, text ="退出", comman=lambda :self.quittop(top))
        end.pack() 
        
    def startMERT(self,n,epoch,k,EditText):
        dataloader = MERTDataLoder(datapath1=self.trainfile, datapath2=self.testfile)
        trainX, trainY = dataloader.loadTrainData()
        testX, testY = dataloader.loadTestData()
        if n:
            n = n
        else:
            n=trainY.shape[0]
        if epoch:
            epoch = epoch
        else:
            epoch=10
        if k:
            k = k
        else:
            k=1
            
        model = MERTModel(n=n,k=k)
        model.fit(trainX=trainX, trainY=trainY, epoch=epoch)
        predictY = model.predict(predictX=testX, predictY=testY)
        EditText.insert(1.0,np.sum(np.power((predictY-testY),2)/(n*testY.shape[0])))
        
    
    def createREBET(self):
        self.win.withdraw()
        top = tk.Toplevel()
        top.title('REBET')
        top.geometry("{}x{}+{}+{}".format(1024, 768, 100, 0))
        filepath1 = tk.Entry(top)
        filepath1.pack()
        file1 = tk.Button(top, text ="训练数据", command=lambda :self.trainfile(filepath1))
        file1.pack() 
        filepath2 = tk.Entry(top)
        filepath2.pack()
        file2 = tk.Button(top, text ="测试数据", command=lambda :self.testfile(filepath2))
        file2.pack() 
        n1 = tk.Label(top, text="类别数(默认值为1)")
        n1.pack()
        n2 = tk.Entry(top)
        n2.pack() 
        epoch1 = tk.Label(top, text="迭代次数(默认值为100)")
        epoch1.pack()
        epoch2 = tk.Entry(top)
        epoch2.pack() 
        k1 = tk.Label(top, text="选择第几个变量与随机效应相关(默认值为1)")
        k1.pack()
        k2 = tk.Entry(top)
        k2.pack() 
        M1 = tk.Label(top, text="迪利克雷聚集参数(默认值为10)")
        M1.pack()
        M2 = tk.Entry(top)
        M2.pack() 
        n = n2.get()
        epoch = epoch2.get()
        k = k2.get()
        M = M2.get()
        start = tk.Button(top, text ="开始", comman=lambda :self.startREBET(n,epoch,k,M,EditText))
        start.pack() 
        EditText = tk.Text(top,width=20,height=10)
        EditText.place(x=512,y=380)
        end = tk.Button(top, text ="退出", comman=lambda :self.quittop(top))
        end.pack() 
        
    def startREBET(self,n,epoch,k,M,EditText):
        dataloader = REBETDataLoder(datapath1=self.trainfile, datapath2=self.testfile)
        trainX, trainY = dataloader.loadTrainData()
        testX, testY = dataloader.loadTestData()
        if n:
            n = n
        else:
            n=trainY.shape[0]
        if epoch:
            epoch = epoch
        else:
            epoch=10
        if k:
            k = k
        else:
            k=1
        if M:
            M = M
        else:
            M=10
            
        model = REBETModel(n=n,k=k,M=M)
        model.fit(trainX=trainX, trainY=trainY, epoch=epoch)
        predictY = model.predict(predictX=testX, predictY=testY)
        EditText.insert(1.0,np.sum(np.power((predictY-testY),2)/(n*testY.shape[0])))
          
    def createIBRT(self):
        self.win.withdraw()
        top = tk.Toplevel()
        top.title('REBET')
        top.geometry("{}x{}+{}+{}".format(1024, 768, 100, 0))
        filepath1 = tk.Entry(top)
        filepath1.pack()
        file1 = tk.Button(top, text ="数据", command=lambda :self.trainfile(filepath1))
        file1.pack() 
        n_iter1 = tk.Label(top, text="迭代次数(默认值为20)")
        n_iter1.pack()
        n_iter2 = tk.Entry(top)
        n_iter2.pack() 
        _gamma1 = tk.Label(top, text="gamma(默认值为0)")
        _gamma1.pack()
        _gamma2 = tk.Entry(top)
        _gamma2.pack() 
        _lambda1 = tk.Label(top, text="lambda(默认值为1)")
        _lambda1.pack()
        _lambda2 = tk.Entry(top)
        _lambda2.pack() 
        max_depth1 = tk.Label(top, text="单颗基本树最大深度(默认值为2)")
        max_depth1.pack()
        max_depth2 = tk.Entry(top)
        max_depth2.pack() 
        n_iter = n_iter2.get()
        _gamma = _gamma2.get()
        _lambda = _lambda2.get()
        max_depth = max_depth2.get()
        start = tk.Button(top, text ="开始", comman=lambda :self.startIBRT(n_iter, _gamma, _lambda, max_depth,EditText))
        start.pack() 
        EditText = tk.Text(top,width=20,height=10)
        EditText.place(x=512,y=380)
        end = tk.Button(top, text ="退出", comman=lambda :self.quittop(top))
        end.pack() 
        
    def startIBRT(self,n_iter, _gamma, _lambda, max_depth,EditText):
        dataloader = IBRTDataLoader(datapath=self.trainfile)
        trainX, trainY = dataloader.loadTrainData()
        testX, testY = dataloader.loadTestData()
           
        if n_iter:
            n_iter = n_iter
        else:
            n_iter=20
        if _gamma:
            _gamma = _gamma
        else:
            _gamma=0
        if _lambda:
            _lambda = _lambda
        else:
            _lambda=1
        if max_depth:
            max_depth = max_depth
        else:
            max_depth=2
        model = IBRTModel(n_iter, _gamma, _lambda, max_depth)
        model.fit(trainX, trainY)
        predictY = model.predict(testX)
        EditText.insert(1.0,mean_absolute_error(testY, predictY))
        
    def createGBM(self):
        self.win.withdraw()
        top = tk.Toplevel()
        top.title('GBM')
        top.geometry("{}x{}+{}+{}".format(1024, 768, 100, 0))
        filepath1 = tk.Entry(top)
        filepath1.pack()
        file1 = tk.Button(top, text ="数据", command=lambda :self.trainfile(filepath1))
        file1.pack() 
        n_estimators1 = tk.Label(top, text="迭代次数(默认值为500)")
        n_estimators1.pack()
        n_estimators2 = tk.Entry(top)
        n_estimators2.pack() 
        max_depth1 = tk.Label(top, text="单个回归估计量的最大深度(默认值为4)")
        max_depth1.pack()
        max_depth2 = tk.Entry(top)
        max_depth2.pack() 
        min_samples_split1 = tk.Label(top, text="分割一个内部节点所需的最小样本数(默认值为5)")
        min_samples_split1.pack()
        min_samples_split2 = tk.Entry(top)
        min_samples_split2.pack() 
        learning_rate1 = tk.Label(top, text="学习器的权重缩减系数(默认值为0.01)")
        learning_rate1.pack()
        learning_rate2 = tk.Entry(top)
        learning_rate2.pack() 
        loss1 = tk.Label(top, text="损失函数(默认值为ls)")
        loss1.pack()
        loss2 = tk.Entry(top)
        loss2.pack() 
        n_estimators = n_estimators2.get()
        max_depth = max_depth2.get()
        min_samples_split = min_samples_split2.get()
        learning_rate = learning_rate2.get()
        loss = loss2.get()
        start = tk.Button(top, text ="开始", comman=lambda :self.startGBM(n_estimators, max_depth, min_samples_split, learning_rate, loss, EditText))
        start.pack() 
        EditText = tk.Text(top,width=20,height=10)
        EditText.place(x=512,y=380)
        end = tk.Button(top, text ="退出", comman=lambda :self.quittop(top))
        end.pack() 
        
    def startGBM(self,n_estimators, max_depth, min_samples_split, learning_rate, loss,EditText):
        dataloader = GBMDataLoader(datapath=self.trainfile)
        trainX, trainY = dataloader.loadTrainData()
        testX, testY = dataloader.loadTestData()
           
        if n_estimators:
            n_estimators = n_estimators
        else:
            n_estimators=500
        if max_depth:
            max_depth = max_depth
        else:
            max_depth=4
        if min_samples_split:
            min_samples_split = min_samples_split
        else:
            min_samples_split=5
        if learning_rate:
            learning_rate = learning_rate
        else:
            learning_rate=0.01
        if loss:
            loss = loss
        else:
            loss='ls'
        params = {'n_estimators': n_estimators,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'learning_rate': learning_rate,
              'loss': 'ls'}
        model = GBMModel(**params)
        model.fit(trainX=trainX, trainY=trainY)
        predictY = model.predict(predictX=testX)
        EditText.insert(1.0,"mse is {:.4f}".format(mean_squared_error(testY, predictY)))
               
    def createSalp(self):
        self.win.withdraw()
        top = tk.Toplevel()
        top.title('Salp')
        top.geometry("{}x{}+{}+{}".format(1024, 768, 100, 0))
        filepath1 = tk.Entry(top)
        filepath1.pack()
        file1 = tk.Button(top, text ="数据", command=lambda :self.trainfile(filepath1))
        file1.pack() 
        start = tk.Button(top, text ="开始", comman=lambda :self.startSalp(EditText))
        start.pack() 
        EditText = tk.Text(top,width=20,height=10)
        EditText.place(x=512,y=380)
        end = tk.Button(top, text ="退出", comman=lambda :self.quittop(top))
        end.pack() 
        
    def startSalp(self,EditText):
        dataloader = SalpDataLoder()
        trainX, trainY = dataloader.loadTrainData()
        testX, testY = dataloader.loadTestData()
           
        model = SVRModel()
        model.fit(trainX=trainX, trainY=trainY)
        predictY = model.predict(predictX=testX)
        EditText.insert(1.0,"mse is {:.4f}".format(mean_squared_error(testY, predictY)))
        
a = simplelearning()
a.run()
