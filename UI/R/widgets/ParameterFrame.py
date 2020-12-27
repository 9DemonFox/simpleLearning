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

        self.parameterValueText = Entry(self, width=16, font=R.font.NormalWeiRuanYaHeiFont(10))
        self.parameterValueText.pack(side=LEFT, padx=5)
        self.parameterValueText.insert(INSERT, parameterStr)

    def setParameter(self, parameterName, parameterStr):
        self.parameterNameLabel["text"] = parameterName
        self.parameterValueText.delete(0, END)
        self.parameterValueText.insert(INSERT, parameterStr)

    def getParameter(self, engOnly=False):
        """
        :param engOnly: 【系数淘汰率 k】=>k
        :return: 参数名字 参数值
        """
        parameterName = self.parameterNameLabel["text"]
        if engOnly:
            parameterName = parameterName[parameterName.find(" ") + 1:]
        return (parameterName, self.parameterValueText.get())
