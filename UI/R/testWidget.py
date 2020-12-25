import tkinter as tk
from tkinter import Frame

from UI.R.widget import SelectLabelsList


class Viewer:
    def __init__(self):
        self.initWindow()

    def run(self):
        self.win.mainloop()

    def initWindow(self):
        self.win = tk.Tk()
        self.win.title("材料腐蚀预测")  # 添加标题
        self.win.geometry("{}x{}+{}+{}".format(960, 640, 100, 0))
        # sF = R.widget.selectFrame("测试", self.win, backgroud="whitesmoke", frontgroud="black")
        # sF.pack()
        # # sF.setSelect()
        # sF.bind("<Button-1>", lambda event: R.widget.selectFrame.SsetSelect(event, sF))
        # sF.bind("<Enter>", lambda event: R.widget.selectFrame.SenterEvent(event, sF))
        # sF.bind("<Leave>", lambda event: R.widget.selectFrame.SleaveEvent(event, sF))
        mainFrame = Frame(self.win, bg="red")

        sFL = SelectLabelsList(mainFrame)
        sFL.initLabelsList()
        sFL.pack()
        sFL.Labelsbind()
        mainFrame.pack()


if __name__ == '__main__':
    v = Viewer()
    v.run()
