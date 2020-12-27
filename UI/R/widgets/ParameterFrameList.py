from tkinter import *

from UI import R
from UI.R.widgets import ParameterFrame


class ParameterFrameList(Frame):
    def __init__(self, parent, parametersList):
        """ 模型列表
        :param parent:
        :param modelsList: ["模型全称","模型简称"]
        :param parametersList: [(parameterName, parameterValue)]
        """
        super().__init__(parent, bg=R.color.UNSelectedColor)
        self.ParameterContainerList = []
        # 申请一个存储Parameter的池子
        self.confirmButton = Button(self, text="确认", width=8, bg=R.color.UNSelectedColor)
        for i in range(10):
            self.ParameterContainerList.append(ParameterFrame(self, "", ""))
        self.repack(parametersList)

    def confirmButtonBind(self):
        pass

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

    def getAllParameters(self):
        """ 返回所有的参数信息
        :return:
        """
        parameterList = []
        for i in range(self.parameterNum):
            parameterList.append(self.ParameterContainerList[i].getParameter(True))
        return parameterList

    def repack(self, parametersList):
        """ 对于新来的参数列表
        :param parametersList:
        :return:
        """
        self.parameterNum = len(parametersList)
        for parameterContainer in self.ParameterContainerList:
            parameterContainer.pack_forget()
        self.confirmButton.pack_forget()

        for (parameterContainer, (parameterName, parameterValue)) in zip(self.ParameterContainerList, parametersList):
            parameterContainer.setParameter(parameterName, parameterValue)
            parameterContainer.pack(pady=5, fill=X)

        self.confirmButton.pack(side=RIGHT)

        self.update()
