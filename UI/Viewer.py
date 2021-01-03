import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk

from UI import R


def tkimg_resized(img, w_box, h_box, keep_ratio=True):
    """对图片进行按比例缩放处理"""
    w, h = img.size

    if keep_ratio:
        if w > h:
            width = w_box
            height = int(h_box * (1.0 * h / w))

        if h >= w:
            height = h_box
            width = int(w_box * (1.0 * w / h))
    else:
        width = w_box
        height = h_box

    img1 = img.resize((width, height), Image.ANTIALIAS)
    tkimg = ImageTk.PhotoImage(img1)
    return tkimg


def image_label(frame, img, width, height, keep_ratio=True):
    """输入图片信息，及尺寸，返回界面组件"""
    if isinstance(img, str):
        _img = Image.open(img)
    else:
        _img = img
    lbl_image = tk.Label(frame, width=width, height=height)

    tk_img = tkimg_resized(_img, width, height, keep_ratio)
    lbl_image.image = tk_img
    lbl_image.config(image=tk_img)
    return lbl_image


class Viewer:
    def __init__(self):
        self.initWindow()

    def run(self):
        self.win.mainloop()

    def initWindow(self):
        self.win = tk.Tk()
        self.win.title("材料腐蚀预测")  # 添加标题
        self.win.geometry("{}x{}+{}+{}".format(960, 640, 100, 0))
        self.body()

    def body(self):
        self.title(self.win).pack(fill=tk.X)
        self.main(self.win).pack(fill=tk.X)
        self.bottom(self.win).pack(fill=tk.X)
        # self.main_left(self.win).pack(side=tk.LEFT, fill=tk.Y, padx=30)

    def bottom(self, parent):
        """ 窗体最下面留空白 """

        frame = tk.Frame(parent, height=10, bg="whitesmoke")
        frame.propagate(True)
        return frame

    def title(self, parent):
        """ 标题栏 """

        def label(frame, text, font):
            return tk.Label(frame, text=text, bg="black", fg="white", height=2, font=font)

        frame = tk.Frame(parent, bg="black")

        label(frame, "机器学习应用平台", R.font.BoldWeiRuanYaHeiFont(16)).pack(side=tk.LEFT, padx=10)

        label1 = label(frame, "回归模型", R.font.NormalWeiRuanYaHeiFont(12))
        label2 = label(frame, "决策分析", R.font.NormalWeiRuanYaHeiFont(12))
        label3 = label(frame, "参数寻优", R.font.NormalWeiRuanYaHeiFont(12))
        self.titleLabelList = [label1, label2, label3]
        label1.pack(side=tk.LEFT, padx=100)
        label2.pack(side=tk.LEFT, padx=0)
        label3.pack(side=tk.LEFT, padx=100)

        label(frame, "", 12).pack(side=tk.RIGHT, padx=20)
        # label(frame, "登录用户", 12).pack(side=tk.RIGHT, padx=20)
        # image_label(frame, "R\\images\\user.png", 40, 40, False).pack(side=tk.RIGHT)

        return frame

    def main(self, parent):
        """ 窗体主体 """

        self.main_frame = tk.Frame(parent, bg="whitesmoke")

        self.main_top(self.main_frame).pack(fill=tk.X, padx=30, pady=15)
        self.main_left(self.main_frame).pack(side=tk.LEFT, fill=tk.Y, padx=30)
        R.widgets.VSeperator(self.main_frame, 30).pack(side=tk.RIGHT, fill=tk.Y)

        # 初始化是使用frame1， main_right_1
        # self.main_right_1(self.main_frame).pack(side=tk.RIGHT, expand=tk.YES, fill=tk.BOTH)

        return self.main_frame

    def main_top(self, parent):
        frame = tk.Frame(parent, bg="white", height=150)

        self.main_top_image_label = image_label(frame, R.image.regression, width=240, height=120,
                                                keep_ratio=False)
        self.main_top_image_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.main_top_middle(frame).pack(side=tk.LEFT)

        frame.propagate(False)
        return frame

    def main_right_1(self, parent):
        """ 在这里初始化3种界面 1.模型选择 2.训练模型 3.模型预测
        :param parent:
        :return:
        """
        self.main_right_frame_1 = tk.Frame(parent, width=200, bg=R.color.SelectedColor)
        # 选择模型
        self.main_right_frame_1_chooseBox = R.widgets.ChooseModelFrame(self.main_right_frame_1, [("", "")])
        self.main_right_frame_1_chooseBox.pack(fill=tk.X, pady=0)
        # 输入变量界面
        self.main_right_frame_1_parameterBox = R.widgets.ParameterFrameList(self.main_right_frame_1, [("", ("", ""))])
        self.main_right_frame_1_parameterBox.pack(fill=tk.X, pady=0)
        # ttk.Button(frame, text="下一步", width=12).pack(anchor=tk.W, padx=112, pady=5)

        return self.main_right_frame_1

    def main_right_2(self, parent):
        """ 在这里初始化3种界面 1.模型选择 2.训练模型
        :param parent:
        :return:
        """
        # 训练模型
        self.main_right_frame_2 = tk.Frame(parent, width=200, bg=R.color.UNSelectedColor)
        self.main_right_frame_2_btnPath = tk.Button(self.main_right_frame_2, text='选择', width=10)
        self.main_right_frame_2_btnTrain = tk.Button(self.main_right_frame_2, text='训练', width=10)
        tk.Label(self.main_right_frame_2, text="数据路径：", bg=R.color.UNSelectedColor).grid(row=0, column=0)
        self.main_right_frame_2_pathEntry = tk.Entry(self.main_right_frame_2, width=45)
        self.main_right_frame_2_pathEntry.grid(row=0, column=1)
        self.main_right_frame_2_btnPath.grid(row=0, column=2)

        tk.Label(self.main_right_frame_2, text="输出结果：", bg=R.color.UNSelectedColor).grid(row=2, column=0)
        self.main_right_frame_2_txtResult = tk.Label(self.main_right_frame_2, bg=R.color.UNSelectedColor, borderwidth=2,
                                                     relief="sunken",
                                                     justify=tk.LEFT,
                                                     width=45, height=15)

        self.main_right_frame_2_txtResult.grid(row=3, column=1)
        self.main_right_frame_2_btnTrain.grid(row=4, column=2)
        # self.main_right_frame_1_chooseBox = R.widgets.ChooseModelFrame(self.main_right_frame_1, [("", "")])
        # self.main_right_frame_1_chooseBox.pack(fill=tk.X, pady=0)
        # self.main_right_frame_1_parameterBox = R.widgets.ParameterFrameList(self.main_right_frame_1, [("", "")])
        # self.main_right_frame_1_parameterBox.pack(fill=tk.X, pady=0)
        # ttk.Button(frame, text="下一步", width=12).pack(anchor=tk.W, padx=112, pady=5)

        return self.main_right_frame_2

    def main_right_4(self, parent):
        """ 在这里初始化3种界面 1.模型选择 2.训练模型 4.预测结果
        :param parent:
        :return:
        """
        # 选择模型
        self.main_right_frame_4 = tk.Frame(parent, width=200, bg=R.color.UNSelectedColor)
        self.main_right_frame_4_btnPath = tk.Button(self.main_right_frame_4, text='选择', width=10)
        self.main_right_frame_4_btnPredict = tk.Button(self.main_right_frame_4, text='预测', width=10)
        tk.Label(self.main_right_frame_4, text="数据路径：", bg=R.color.UNSelectedColor).grid(row=0, column=0)
        self.main_right_frame_4_pathEntry = tk.Entry(self.main_right_frame_4, width=45)
        self.main_right_frame_4_pathEntry.grid(row=0, column=1)
        self.main_right_frame_4_btnPath.grid(row=0, column=2)

        tk.Label(self.main_right_frame_4, text="输出结果：", bg=R.color.UNSelectedColor).grid(row=2, column=0)
        self.main_right_frame_4_txtResult = tk.Label(self.main_right_frame_4, bg=R.color.UNSelectedColor, borderwidth=2,
                                                     relief="sunken",
                                                     justify=tk.LEFT,
                                                     width=45, height=15)

        self.main_right_frame_4_txtResult.grid(row=3, column=1)
        self.main_right_frame_4_btnPredict.grid(row=4, column=2)
        # self.main_right_frame_1_chooseBox = R.widgets.ChooseModelFrame(self.main_right_frame_1, [("", "")])
        # self.main_right_frame_1_chooseBox.pack(fill=tk.X, pady=0)
        # self.main_right_frame_1_parameterBox = R.widgets.ParameterFrameList(self.main_right_frame_1, [("", "")])
        # self.main_right_frame_1_parameterBox.pack(fill=tk.X, pady=0)
        # ttk.Button(frame, text="下一步", width=12).pack(anchor=tk.W, padx=112, pady=5)

        return self.main_right_frame_4

    def layoutConfigModel(self, modelName, parameters: dict):
        """ 对模型配置布局
        :return:
        """
        pass

    def main_top_middle(self, parent):
        str1 = "多元线性回归可表示为y=β0+β1*x+εi，式中，β0，β1，…，βp"
        str2 = "是p+1个待估计的参数，εi是相互独立且服从同一正态分布N(0,σ2)的随机变量"

        def label(frame, text):
            return tk.Label(frame, bg="white", fg="gray", text=text, font=R.font.NormalWeiRuanYaHeiFont(12))

        frame = tk.Frame(parent, bg="white")

        self.main_top_middle_top(frame).pack(anchor=tk.NW)
        self.main_top_middle_label1 = label(frame, str1)
        self.main_top_middle_label2 = label(frame, str2)
        self.main_top_middle_label1.pack(anchor=tk.W, padx=10, pady=2)
        self.main_top_middle_label2.pack(anchor=tk.W, padx=10)

        return frame

    def main_top_middle_top(self, parent):
        def label(frame, text, font=R.font.NormalWeiRuanYaHeiFont(12), fg="blue"):
            return tk.Label(frame, text=text, bg="white", fg=fg, font=font)

        frame = tk.Frame(parent, bg="white")

        self.main_top_middle_top_label = label(frame, "回归模型", R.font.BoldWeiRuanYaHeiFont(20), "black")
        self.main_top_middle_top_label.pack(side=tk.LEFT, padx=10)
        label(frame, "操作文档").pack(side=tk.LEFT, padx=10)
        label(frame, "教学视频").pack(side=tk.LEFT, padx=10)
        label(frame, "常见问题").pack(side=tk.LEFT, padx=10)

        return frame

    def main_left(self, parent):
        self.main_left_frame = tk.Frame(parent, bg="white")
        self.main_left_chooseStepList = None
        return self.main_left_frame

    def trainfile(self, filepath1):
        self.trainfile = filedialog.askopenfilename()
        filepath1.insert(0, self.trainfile)

    def testfile(self, filepath2):
        self.testfile = filedialog.askopenfilename()
        filepath2.insert(0, self.testfile)


if __name__ == '__main__':
    view = Viewer()
    view.run()
