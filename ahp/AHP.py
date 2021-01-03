import copy
import re
import uuid

import numpy as np

from ahp.tree import Tree
from model import Model

MATRIX_FLAG = "matrix_"  # 矩阵标识开头
from abc import ABC, abstractmethod


class Method(ABC):
    """Method

    Abstract class to provide a common interface for various methods.
    """

    @abstractmethod
    def estimate(self, preference_matrix):
        """Estimate the priority from the provided preference matrix

        Args:
            preference_matrix (np.array): A two (nxn) dimensional reciprocal array.

        Returns:
            nx1 np.array of priorities
        """
        pass

    @staticmethod
    def _evaluate_consistency(matrix):
        """
        矩阵的一致性估计
        """
        A = matrix
        """
             https://zhuanlan.zhihu.com/p/93109898
             https: // jingyan.baidu.com / article / cbf0e500eb95582eaa28932b.html
        """
        n = A.shape[0]
        if n == 1:
            return True;  # 如果是1阶矩阵，一致性通过

        columns_sum = A.sum(axis=0)  # 结果为行 [[1,2,3],[1,2,3],[1,2,3]]==>[3,6,9]
        column_norm_matrix = A / columns_sum  # [0.333,0.333,0.333]。。（3）
        row_sum_vec = column_norm_matrix.sum(axis=1)  # 行和向量
        sigma_row_sum_vec = row_sum_vec.sum()
        W = row_sum_vec / sigma_row_sum_vec
        W = W.transpose()
        AW = np.matmul(A, W)
        sigma = 0
        for i in range(n):
            sigma += AW[i] / W[i]
        Lamda_MAX = (1 / n) * sigma
        CI = (Lamda_MAX - n) / (n - 1)  # CI=0时完全一致
        if RANDOM_INDICES[n - 1] == 0:
            return True
        else:
            CR = CI / RANDOM_INDICES[n - 1]
            # TODO 此处是否取绝对值
            return (abs(CR) < 0.1)

    @staticmethod
    def _check_matrix(matrix):
        """
        偏好矩阵是否合法
        :param matrix:
        :return:
        """
        width, height = matrix.shape
        assert width == height, "比较矩阵必须是方阵"
        # TODO 修改矩阵检测方法 对于等于1的矩阵就取权重为1，对于二维矩阵就取比例或者比例的某个系数
        assert width > 0, "矩阵维度不能为空"
        for i in range(width):
            for j in range(height):
                if i == j:
                    assert matrix[i, j] == 1, "比较矩阵对角线必须为1"
                else:
                    # matrix[i,j]*matrix[j,i]==1 数值计算有误差 1/3 3 那么3*0.33=0.99
                    assert abs(1 - matrix[i, j] * matrix[j, i]) <= 0.011, "aij和aji必须为倒数关系"
        assert Method._evaluate_consistency(matrix), "一致性检测不通过，请修改您的偏好矩阵"


RANDOM_INDICES = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51]

from numpy.linalg import eig


class EigenvalueMethod(Method):
    """Eigenvalue based priority estimation method
    """

    def estimate(self, preference_matrix):
        """
        返回一个等于维度的向量，每个元素代表权重
        :param preference_matrix:
        :return:
        """
        super()._check_matrix(preference_matrix)  # 一致性检测
        A = preference_matrix
        """
             https://zhuanlan.zhihu.com/p/93109898
             https: // jingyan.baidu.com / article / cbf0e500eb95582eaa28932b.html
        """
        eigs, vectors = eig(preference_matrix)
        n = A.shape[0]
        if eigs[eigs != 0].shape[0] <= 1:
            super()._check_matrix(preference_matrix)
            row_sums = np.sum(preference_matrix, axis=1)
            total_sum = np.sum(row_sums)
            return row_sums / total_sum
        else:
            width = preference_matrix.shape[0]
            _, vectors = eig(preference_matrix)
            real_vector = np.real([vec for vec in np.transpose(vectors) if not np.all(np.imag(vec))][:1])
            sum_vector = np.sum(real_vector)
            self._evaluate_consistency(preference_matrix)
            W = np.around(real_vector, decimals=3)[0] / sum_vector
            return W


class ApproximateMethod(Method):
    """Approximate priority estimation method
    """

    def estimate(self, preference_matrix):
        # 可以估计sample中的例子
        super()._check_matrix(preference_matrix)
        row_sums = np.sum(preference_matrix, axis=1)
        total_sum = np.sum(row_sums)
        return row_sums / total_sum


class GeometricMethod(Method):
    """Geometric priority estimation method
    """

    def estimate(self, preference_matrix):
        super()._check_matrix(preference_matrix)
        height = preference_matrix.shape[1]
        vec = np.prod(preference_matrix, axis=1) ** 0.5 / height
        return vec / np.sum(vec)


class localModel():
    """
    # TODO 设置偏好矩阵行列的规则
    """
    __model = {}
    func_dict = {}  # 映射方法的字典
    original_content = []  # 存储原数据

    def __init__(self):
        pass

    def __set_model_tree(self):
        """
        根据模型构造出树
        :return: 构造出的层次模型
        """
        m = self.__model
        method_class = globals()[m['method'].capitalize() + "Method"]

        method = method_class()
        # 构造得分dict
        criteria_m = np.asarray(m['preferenceMatrices']['criteria']).astype(np.float)
        try:
            criteria_w = method.estimate(criteria_m)
        except Exception as e:
            raise Exception("【准则层】" + " " + e.args[0])
        t = Tree()
        root = t.create_node(identifier=m['name'], data=1)  # 根节点 0层
        for i in range(len(m['criteria'])):  # 创建第1层
            t.create_node(identifier=m['criteria'][i], data=criteria_w[i], parent=m['name'])
        # 将子准则层的矩阵都提取出来
        subpms = copy.deepcopy(m['preferenceMatrices'])
        del subpms['criteria']
        for item in subpms:
            parent_tag = re.sub('subCriteria:', '', item)
            t_matrix = np.asarray(subpms[item]).astype(np.float)
            try:  # 返回出错的矩阵名字
                weights = method.estimate(t_matrix)
            except Exception as e:
                raise Exception("【" + parent_tag + "】" + " " + e.args[0])
            zips = zip(weights, m['subCriteria'][parent_tag])
            for (weight, tag) in zips:
                t.create_node(identifier=tag + str(uuid.uuid4()), data=weight, parent=parent_tag, tag=tag)
        for node in t.all_nodes():
            node.data = round(node.data, 3)
        self.__model_tree = t
        return t

    def set_model_from_model(self, m):
        self.__model = m
        self.__set_model_tree()

    def __narrow(self, matrix, coeff):
        """
            将矩阵中的差异缩小
        """
        # 先将大于1的元素缩小
        i, j = 0, 0
        while i < len(matrix):
            line = matrix[i]
            j = 0
            while j < len(line):
                if float(matrix[i][j]) > 1:
                    matrix[i][j] = 1 + (float(matrix[i][j]) - 1) * coeff
                elif float(matrix[i][j]) == 1:
                    matrix[i][j] = float(matrix[i][j])
                j += 1
            i += 1
        i, j = 0, 0
        while i < len(matrix):
            line = matrix[i]
            j = 0
            while j < i:  # 下三角元素
                if float(matrix[i][j]) < 1:
                    matrix[i][j] = round(1 / float(matrix[j][i]), 2)
                j += 1
            i += 1

    def narrow_difference(self, coeff=0.5):
        # 减小间距
        # TODO coeff取值过大会导致比较矩阵不通过
        m = self.__model
        for key in m["preferenceMatrices"].keys():
            temp_m = m["preferenceMatrices"][key]
            temp_m = self.__narrow(temp_m, coeff)
        self.set_model_from_model(m)

    def get_model(self):
        return self.__model

    def get_model_tree(self):
        return self.__model_tree


def creat_model(m: dict):
    """
    建立模型
    """
    model = localModel()
    model.set_model_from_model(m)
    return model


class AHPModel(Model):

    def __init__(self):
        pass

    def fit(self, **kwargs):
        trainX = kwargs["trainX"]
        # trainY = kwargs["trainY"]
        self.model = creat_model(trainX)

    def predict(self, **kwargs):
        predictX = kwargs["predictX"]
        self.model = creat_model(predictX)
        return self.model.get_model_tree().getShow()


if __name__ == '__main__':
    trainX = {'criteria': ['子准则层1', '子准则层2', '子准则层3', '子准则层4', '子准则层5'],
              'method': 'eigenvalue',
              'name': '画像',
              'preferenceMatrices': {'criteria': [['1', '1', '2', '1', '2'],  # 准则层
                                                  ['1', '1', '2', '1', '1'],
                                                  ['0.5', '0.5', '1', '1', '1'],
                                                  ['1', '1', '1', '1', '2'],
                                                  ['0.5', '1', '1', '0.5', '1']],
                                     'subCriteria:子准则层1': [['1', '1', '1', '1'],
                                                           ['1', '1', '1', '1'],
                                                           ['1', '1', '1', '1'],
                                                           ['1', '1', '1', '1']],
                                     'subCriteria:子准则层2': [['1', '2'], ['0.5', '1']],
                                     'subCriteria:子准则层3': [['1']],
                                     'subCriteria:子准则层4': [['1', '3'], ['0.33', '1']],
                                     'subCriteria:子准则层5': [['1', '2', '3'],
                                                           ['0.5', '1', '2'],
                                                           ['0.33', '0.5', '1']]},
              'subCriteria': {'子准则层1': ['兽残不合格量', '毒素不合格量', '污染物不合格量', '重金属不合格量'],  # 决策层
                              '子准则层2': ['U1占比', '综合合格率'],
                              '子准则层3': ['周期内预警触发次数*'],
                              '子准则层4': ['牧场整改率', '牧场食品安全评审结果'],
                              '子准则层5': ['主要理化指标Cpk*', '体细胞Cpk*', '微生物Cpk*']}}
    model = AHPModel()
    model.fit(trainX=trainX)
    predictY = model.predict()
    print(predictY)
