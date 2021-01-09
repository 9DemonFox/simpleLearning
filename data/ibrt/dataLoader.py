from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from data.dataLoader import DataLoader


DATAPATH = "test.xlsx"

def fra_Data(data, fra):  # fra为正整数
    ret = data.copy()
    while fra > 1:
        ret = np.concatenate((ret, data), axis=0)
        fra -= 1
    return ret

class IBRTDataLoader(DataLoader):
    def __init__(self, datapath=DATAPATH):
        data = pd.read_excel(datapath)
        valid_data = np.array(data.iloc[1:])
        valid_data = fra_Data(valid_data, 3)
        X = valid_data[:, 1:-1]
        y = valid_data[:, -1]

        self.trainX, self.testX = train_test_split(X, test_size=0.1, random_state=7)
        self.trainY, self.testY = train_test_split(y, test_size=0.1, random_state=7)

    def loadTrainData(self):
        return self.trainX, self.trainY

    def loadTestData(self):
        return self.testX, self.testY