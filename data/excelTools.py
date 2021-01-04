import numpy
import pandas


class ExcelTool:

    @staticmethod
    def saveXY2Excel(x, y, path="SALP_TRAIN_DATA.xlsx"):
        """
        :param self:
        :param x:
        :param y:
        :param path: 输出excel路径
        :return:
        """
        xy = numpy.insert(x, 0, values=y, axis=1)
        df = pandas.DataFrame.from_records(xy)
        columns = ["y"]
        columns_x = ["x" + str(i) for i in range(x.shape[1])]
        columns.extend(columns_x)
        df.columns = columns
        df.to_excel(path)

    @staticmethod
    def saveX2Excel(x, path="SALP_PREDICT_DATA.xlsx"):
        """ 存储预测数据集
        :param x:
        :param path:
        :return:
        """
        columns = ["x" + str(i) for i in range(x.shape[1])]
        df = pandas.DataFrame.from_records(x)
        df.columns = columns
        df.to_excel(path)

    @staticmethod
    def loadExcelData(data_path):
        """
        :param data_path: excel数据 第1列为Y
        :return:
        """
        df = pandas.read_excel(data_path, index_col=0)
        y = df.values[:, 0]
        x = df.values[:, 1:]
        return x, y
