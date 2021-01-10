import numpy
import pandas

from data.ibrt.dataLoader import IBRTDataLoader


class excelLoader():
    def __init__(self):
        pass

    def loadTrainData(self):
        pass

    def loadPredicData(self):
        pass

    def loadTrainData(self):
        pass


    def saveXY2Excel(self, x, y, path):
        xy = numpy.insert(x, 0, values=y, axis=1)
        df = pandas.DataFrame.from_records(xy)
        print(df)
        columns = ["y"]
        columns_x = ["x" + str(i) for i in range(trainX.shape[1])]
        columns.extend(columns_x)
        df.columns = columns
        df.to_excel(path)

    def saveX2Excel(self, x, path="SALP_PREDICT_DATA.xlsx"):
        """ 存储预测数据集
        :param x:
        :param path:
        :return:
        """
        columns = ["x" + str(i) for i in range(x.shape[1])]
        df = pandas.DataFrame.from_records(x)
        df.columns = columns
        df.to_excel(path)

if __name__ == '__main__' :
    train_datapath = "ibrt_train_data.xlsx"
    test_datapath = "ibrt_test_data.xlsx"
    predict_datapath = "ibrt_predict_data.xlsx"
    dataloader = IBRTDataLoader()
    trainX, trainY = dataloader.loadTrainData(train_path=train_datapath)
    testX, testY = dataloader.loadTestData(test_path=test_datapath)
    excel = excelLoader()
    excel.saveXY2Excel(x=trainX, y=trainY, path="IBRT_TRAIN_DATA.xlsx")
    excel.saveXY2Excel(testX, testY, "IBRT_TEST_DATA.xlsx")
    print(testX[0].reshape(1,-1))
    excel.saveX2Excel(testX[0].reshape(1,-1), "IBRT_PREDICT_DATA.xlsx")

if __name__ == '__main__' and False:
    df = pandas.read_excel("./test.xlsx", index_col=0)
    y = df.values[:, 0]
    x = df.values[:, 1:]
    print(x.shape, y.shape)
