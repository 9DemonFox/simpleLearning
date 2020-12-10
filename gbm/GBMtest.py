import unittest
from sklearn.metrics import mean_squared_error

from gbm.GBM import GBMModel
from data.gbm.dataLoader import GBMDataLoader

if __name__ == "__main__":
    # Quadric 损失函数 (y-f(x))^2 / 2 -> ls
    # Laplace 损失函数 abs(y-f(x)) -> lad
    params = {'n_estimators': 500,
              'max_depth': 4,
              'min_samples_split': 5,
              'learning_rate': 0.01,
              # 'loss': 'lad',
              'loss': 'ls'}
    gbm_reg = GBMModel(**params)
    gbm_loader = GBMDataLoader(datapath="../data/gbm/oil_field_data_for_gbm.xlsx")

    trainX, trainY = gbm_loader.loadTrainData()
    testX, testY = gbm_loader.loadTestData()

    gbm_reg.fit(trainX=trainX, trainY=trainY)
    predictY = gbm_reg.predict(predictX=testX)
    print("mse is {:.4f}".format(mean_squared_error(testY, predictY)))
