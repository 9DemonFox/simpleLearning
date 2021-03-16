from tkinter import *

from UI import R
from UI.R.widgets import ParameterFrame, ParameterSelectFrame


class ParameterFrameList(Frame):
    ParameterContainerNum = 10  # 配置ParameterContainer最大个数

    def __init__(self, parent, parametersList):
        """ 模型列表
        :param parent:
        :param modelsList: ["模型全称","模型简称"]
        :param parametersList: [(parameterName, parameterValue)]
        """
        super().__init__(parent, bg=R.color.UNSelectedColor)
        self.ParameterContainerList = []
        self.ParameterSelectContainerList = []
        self.Parameters = []
        # 申请一个存储Parameter的池子
        self.confirmButton = Button(self, text="确认", width=8, bg=R.color.UNSelectedColor)
        for i in range(10):
            self.ParameterContainerList.append(ParameterFrame(self, "", ""))
        for i in range(5):
            self.ParameterSelectContainerList.append(ParameterSelectFrame(self, "", []))
        self.repack(parametersList)

    def setParametersLayout(self):
        """ 设置参数布局
        :return:
        """

    def bind(self, recall, C):
        """
        :param recall: 回调函数
        :param C: 控制器
        :return:
        """
        self.confirmButton.bind("<Button-1>", lambda event: recall(event, C))

    def getParametersDict(self):
        """ 返回所有的参数信息
        :return:
        {
            "k":"10"
        }
        """
        parameterDict = {}
        for i in range(self.parameterNum):
            parameterName, parameterValue = self.Parameters[i].getParameter()
            parameterDict[parameterName] = parameterValue
        return parameterDict

    def repack(self, parametersList):
        """ 对于新来的参数列表
        :param parametersList:
        :return:
        """
        self.Parameters.clear()
        self.parameterNum = len(parametersList)
        for parameterContainer in self.ParameterContainerList:
            parameterContainer.pack_forget()
        for parameterSelectContainer in self.ParameterSelectContainerList:
            parameterSelectContainer.pack_forget()
        self.confirmButton.pack_forget()
        strParaNum, listParaNum = 0, 0
        for (parameterName, (parameterNameCn, parameterValue)) in parametersList:
            print(parameterValue, type(parameterValue))
            if type(parameterValue) == str or type(parameterValue) == int or type(parameterValue) == float:
                self.ParameterContainerList[strParaNum].setParameter(parameterName, parameterNameCn, parameterValue)
                self.ParameterContainerList[strParaNum].pack(pady=5, fill=X)
                self.Parameters.append(self.ParameterContainerList[strParaNum])
                strParaNum += 1
            elif type(parameterValue) == list:
                self.ParameterSelectContainerList[listParaNum].setParameter(parameterName, parameterNameCn,
                                                                            parameterValue)
                self.ParameterSelectContainerList[listParaNum].pack(pady=5, fill=X)
                self.Parameters.append(self.ParameterSelectContainerList[listParaNum])
                listParaNum += 1

        self.confirmButton.pack(side=RIGHT)

        self.update()
