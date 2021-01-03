from tkinter import *

from UI import R


def space(n):
    s = " "
    r = ""
    for i in range(n):
        r += s
    return r


class ParameterFrame(Frame):
    """ 包含三个部分组成
    """

    def __init__(self, parent, parameterName, parameterStr):
        super().__init__(parent, bg=R.color.UNSelectedColor)
        Label(self, text="*", fg="red", bg=R.color.UNSelectedColor).pack(side=LEFT, pady=5)
        if len(parameterName) > 16:
            parameterName = parameterName[:16] + "..."
        self.parameterNameLabel = Label(self, text=parameterName, anchor=E, bg=R.color.UNSelectedColor,
                                        font=R.font.NormalWeiRuanYaHeiFont(10),
                                        width=12)
        self.parameterNameLabel.pack(side=LEFT, padx=0)

        self.parameterNameLabelCN = Label(self, text=parameterName, anchor=E, bg=R.color.UNSelectedColor,
                                          font=R.font.NormalWeiRuanYaHeiFont(10),
                                          width=12)
        self.parameterNameLabelCN.pack(side=LEFT, padx=0)

        self.parameterValueEntry = Entry(self, width=16, font=R.font.NormalWeiRuanYaHeiFont(10))
        self.parameterValueEntry.pack(side=LEFT, padx=5)
        self.parameterValueEntry.insert(INSERT, parameterStr)

    def setParameter(self, parameterName, parameterNameCN, parameterStr):
        self.parameterNameLabel["text"] = parameterName
        self.parameterNameLabelCN["text"] = parameterNameCN
        self.parameterValueEntry.delete(0, END)
        self.parameterValueEntry.insert(INSERT, parameterStr)

    def getParameter(self):
        """
        :param engOnly: 【系数淘汰率 k】=>k
        :return: 参数名字 参数值
        """
        parameterName = self.parameterNameLabel["text"]
        return (parameterName, self.parameterValueEntry.get())
