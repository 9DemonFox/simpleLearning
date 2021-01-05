from tkinter import *


class VSeperator(Frame):
    """ 垂直分割线
    """

    def __init__(self, parent, width, bg="whitesmoke"):  # width 单位为像素值
        super().__init__(parent, width=width, bg=bg)

    def pack(self, side=LEFT, fill=Y):
        super().pack(side=side, fill=fill)
