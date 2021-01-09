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


if __name__ == '__main__' :
    dataloader = IBRTDataLoader()
    trainX, trainY = dataloader.loadTrainData()
    testX, testY = dataloader.loadTestData()
    excel = excelLoader()
    excel.saveXY2Excel(trainX, trainY, "IBRT_TRAIN_DATA.xlsx")
    excel.saveXY2Excel(testX, testY, "IBRT_TEST_DATA.xlsx")

if __name__ == '__main__' and False:
    df = pandas.read_excel("./test.xlsx", index_col=0)
    y = df.values[:, 0]
    x = df.values[:, 1:]
    print(x.shape, y.shape)
