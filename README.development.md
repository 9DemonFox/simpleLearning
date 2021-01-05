# 1. 代码规范

**参考ahp文件夹下AHPModel**

1. 所有模型新建文件夹为模型名字
2. 所有模型需要继承model类并且实现fit predict方法
3. 训练数据为trainX, trainY, 预测为predictX, predictY,没有的不写
4. 如果有数据从外部加载，在data下建立文件夹存数据
5. 对每个模型写测试用例

**参考data/salp/dataLoder，测试类class SAPTestCase(unittest.TestCase)中def testDataLoder(self):**

1. 数据加载必须要继承data/dataLoader.py 中的DataLoder类实现其中的虚函数，没有的使用pass

# 2. 运行

运行modelTest.py,可以运行单个和多个模型

# 3. 引入外部的包

在此处写下版本和命令

```
pip install numpy=1.19.2
pip install scipy=1.4.1
pip install sklearn
```

# 4. 提交！！！

### 新建本地分支myBranch，切换到本地分支myBranch，然后推送到远程，请求合并
### 测试用例必须全部通过



# 会议纪要

## 2021年1月3日

```
1. 模型参数都放在__init__(),fit和predict不要参数
2. 在配置UIConfig增加模型配置
3.
	def fitForUI(self, **kwargs):
        """ 返回结果到前端
        :return:
        """
        self.fit(**kwargs)
        # 返回结果为字典形式
        excludeFeatures, coefs = self.fit(**kwargs)
        returnDic = {
            "排除特征": str(excludeFeatures),
            "系数": str(coefs)
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
4. DataLoader

	def loadTrainData(self, **kwargs):
        assert "train_path" in kwargs.keys()
        trainX, trainY = self.__loadExcelData(kwargs.get("train_path"))
        return trainX, trainY

    def loadTestData(self, **kwargs):
        assert "test_path" in kwargs.keys()
        testX, testY = self.__loadExcelData(kwargs.get("test_path"))
        return testX, testY
    或者参考ahp模型从文本加载数据集
    def loadPredictData(self, **kwargs):
        """
        :param predict_path 预测的输入
        :return: 加载数据
        """
        assert "predict_path" in kwargs.keys()
        with open(kwargs.get("predict_path"), "r", encoding="utf-8")as f:
            text = f.read()
        dic = eval(text)
        return dic
```

