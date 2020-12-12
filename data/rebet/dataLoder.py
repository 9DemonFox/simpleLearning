import numpy as np
from data.dataLoder import DataLoder



class REBETDataLoder(DataLoder):
    def __init__(self, datapath1,datapath2):
        data_train = np.loadtxt(open(datapath1,"rb"),delimiter=",",skiprows=0)
        data_test = np.loadtxt(open(datapath2,"rb"),delimiter=",",skiprows=0)
        self.trainY = data_train[:,0:1]
        self.trainX = data_train[:,1:]

        self.predictY = data_test[:,0:1]
        self.predictX = data_test[:,1:]

    def loadTrainData(self):
        return self.trainX, self.trainY

    def loadTestData(self):
        return self.predictX, self.predictY


if __name__ == "__main__":

    datapath1 = "./data_train.csv"
    datapath2 = "./data_test.csv"
    dataloder = REBETDataLoder(datapath1,datapath2)
    data_train = dataloder.loadTrainData()
    data_test = dataloder.loadTrainData()
    trainY = data_train[:,0:1]
    trainX = data_train[:,1:]

    predictY = data_test[:,0:1]
    predictX = data_test[:,1:]