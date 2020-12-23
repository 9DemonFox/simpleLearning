import numpy
import pandas

from data.salp.dataLoder import SalpDataLoader


class excelLoader():
    def __init__(self):
        pass

    def loadTrainData(self):
        pass

    def loadPredicData(self):
        pass

    def loadTrainData(self):
        pass

    def saveXY2Excel(self, x, y, path="SALP_TRAIN_DATA.xlsx"):
        xy = numpy.insert(x, 0, values=y, axis=1)
        df = pandas.DataFrame.from_records(xy)
        columns = ["x" + str(i) for i in range(trainX.shape[1])]
        columns.append("y")
        df.columns = columns
        df.to_excel(path)


if __name__ == '__main__':
    dataloader = SalpDataLoader("./SALP_DATA.npy")
    trainX, trainY = dataloader.loadTrainData()
    testX, testY = dataloader.loadTestData()
    excel = excelLoader()
    excel.saveXY2Excel(trainX, trainY, "SALP_TRAIN_DATA.xlsx")
    excel.saveXY2Excel(testX, testY, "SALP_TEST_DATA.xlsx")
