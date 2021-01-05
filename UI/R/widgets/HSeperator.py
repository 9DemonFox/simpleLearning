from tkinter import *


class HSeperator(Frame):  # height 单位为像素值
    """水平分割线
    """

    def __init__(self, parent, height, bg="whitesmoke"):  # width 单位为像素值
        super().__init__(parent, height=height, bg=bg)

    def pack(self):
        super().pack(fill=X)
