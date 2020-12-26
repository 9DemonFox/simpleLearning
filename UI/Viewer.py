import tkinter as tk
from tkinter import filedialog
from tkinter import font as tkFont
from tkinter import ttk

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


def _font(fname="微软雅黑", size=12, bold=tkFont.NORMAL):
    """设置字体"""
    ft = tkFont.Font(family=fname, size=size, weight=bold)
    return ft


def _ft(size=12, bold=False):
    """极简字体设置函数"""
    if bold:
        return _font(size=size, bold=tkFont.BOLD)
    else:
        return _font(size=size, bold=tkFont.NORMAL)


def h_seperator(parent, height=2):  # height 单位为像素值
    """水平分割线, 水平填充 """
    tk.Frame(parent, height=height, bg="whitesmoke").pack(fill=tk.X)


def v_seperator(parent, width, bg="whitesmoke"):  # width 单位为像素值
    """垂直分割线 , fill=tk.Y, 但如何定位不确定，直接返回对象，由容器决定 """
    frame = tk.Frame(parent, width=width, bg=bg)
    return frame


def selectedLabel(parent, label):
    """ 选中某个按键
    :param parent:
    :param label:
    :return:
    """
    f1 = tk.Frame(parent, bg="whitesmoke")
    v_seperator(f1, width=5, bg="blue").pack(side=tk.LEFT, fill=tk.Y)
    label(f1, "创建模型", bg="whitesmoke").pack(side=tk.LEFT, anchor=tk.W, padx=35, pady=5)
    f1.pack(fill=tk.X)
    pass


class EventHandler:
    @classmethod
    def unselectLabel(event, seperator):
        label = event.widget
        label["backgroud"] = "white"
        seperator["backgroud"] = "white"


class SelectLabelsList:
    text2LabelDic = {}
    labels = []  # 记录的按键
    selectFlag = []  # 选中的标签

    def __init__(self):
        pass


class Viewer:
    def __init__(self):
        self.initWindow()
        self.initMenubar()

    def run(self):
        self.win.mainloop()

    def initMenubar(self):
        """ 创建菜单栏
        :return:
        """
        self.menubar = tk.Menu(self.win)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)  # tearoff意为下拉
        self.menubar.add_cascade(label='模型选择', menu=self.filemenu)
        self.win.config(menu=self.menubar)  # 加上这代码，才能将菜单栏显示

    def initWindow(self):
        self.win = tk.Tk()
        self.win.title("材料腐蚀预测")  # 添加标题
        self.win.geometry("{}x{}+{}+{}".format(960, 640, 100, 0))
        self.body()

    def body(self):
        """
    title title title
    mainLeft mainTop
        """
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

        def label(frame, text, size, bold=False):
            return tk.Label(frame, text=text, bg="black", fg="white", height=2, font=_ft(size, bold))

        frame = tk.Frame(parent, bg="black")

        label(frame, "机器学习应用平台", 16, True).pack(side=tk.LEFT, padx=10)

        label1 = label(frame, "回归模型", 12)
        label2 = label(frame, "决策分析", 12)
        label3 = label(frame, "参数寻优", 12)
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

        frame = tk.Frame(parent, bg="whitesmoke")

        self.main_top(frame).pack(fill=tk.X, padx=30, pady=15)
        self.main_left(frame).pack(side=tk.LEFT, fill=tk.Y, padx=30)
        v_seperator(frame, 30).pack(side=tk.RIGHT, fill=tk.Y)
        self.main_right(frame).pack(side=tk.RIGHT, expand=tk.YES, fill=tk.BOTH)

        return frame

    def main_top(self, parent):
        def label(frame, text, size=12):
            return tk.Label(frame, bg="white", fg="gray", text=text, font=_ft(size))

        frame = tk.Frame(parent, bg="white", height=150)

        self.main_top_image_label = image_label(frame, R.image.regression, width=240, height=120,
                                                keep_ratio=False)
        self.main_top_image_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.main_top_middle(frame).pack(side=tk.LEFT)

        frame.propagate(False)
        return frame

    def main_right(self, parent):
        def label(frame, text, size=10, bold=False, fg="black"):
            return tk.Label(frame, text=text, bg="white", fg=fg, font=_ft(size, bold))

        def space(n):
            s = " "
            r = ""
            for i in range(n):
                r += s
            return r

        frame = tk.Frame(parent, width=200, bg="white")

        label(frame, "创建模型", 12, True).pack(anchor=tk.W, padx=20, pady=5)

        h_seperator(frame)

        f1 = tk.Frame(frame, bg="white")
        label(f1, space(8) + "模型类别:").pack(side=tk.LEFT, pady=5)
        label(f1, "线性回归").pack(side=tk.LEFT, padx=20)
        f1.pack(fill=tk.X)

        f2 = tk.Frame(frame, bg="white")
        label(f2, space(5) + "*", fg="red").pack(side=tk.LEFT, pady=5)
        label(f2, "模型名称:").pack(side=tk.LEFT)
        tk.Entry(f2, bg="white", font=_ft(10), width=25).pack(side=tk.LEFT, padx=20)
        f2.pack(fill=tk.X)

        f3 = tk.Frame(frame, bg="white")
        label(f3, space(5) + "*", fg="red").pack(side=tk.LEFT, pady=5)
        label(f3, "联系方式:").pack(side=tk.LEFT)
        tk.Entry(f3, bg="white", font=_ft(10), width=25).pack(side=tk.LEFT, padx=20)
        f3.pack(fill=tk.X)

        f4 = tk.Frame(frame, bg="white")
        label(f4, space(5) + "*", fg="red").pack(side=tk.LEFT, anchor=tk.N, pady=5)
        label(f4, "功能描述:").pack(side=tk.LEFT, anchor=tk.N, pady=5)
        tk.Text(f4, bg="white", font=_ft(10), height=10, width=40).pack(side=tk.LEFT, padx=20, pady=5)
        f4.pack(fill=tk.X)

        ttk.Button(frame, text="下一步", width=12).pack(anchor=tk.W, padx=112, pady=5)

        return frame

    def main_top_middle(self, parent):
        str1 = "多元线性回归可表示为y=β0+β1*x+εi，式中，β0，β1，…，βp"
        str2 = "是p+1个待估计的参数，εi是相互独立且服从同一正态分布N(0,σ2)的随机变量"

        def label(frame, text):
            return tk.Label(frame, bg="white", fg="gray", text=text, font=_ft(12))

        frame = tk.Frame(parent, bg="white")

        self.main_top_middle_top(frame).pack(anchor=tk.NW)
        self.main_top_middle_label1 = label(frame, str1)
        self.main_top_middle_label2 = label(frame, str2)
        self.main_top_middle_label1.pack(anchor=tk.W, padx=10, pady=2)
        self.main_top_middle_label2.pack(anchor=tk.W, padx=10)

        return frame

    def main_top_middle_top(self, parent):
        def label(frame, text, size=12, bold=True, fg="blue"):
            return tk.Label(frame, text=text, bg="white", fg=fg, font=_ft(size, bold))

        frame = tk.Frame(parent, bg="white")

        self.main_top_middle_top_label = label(frame, "回归模型", 20, True, "black")
        self.main_top_middle_top_label.pack(side=tk.LEFT, padx=10)
        label(frame, "操作文档").pack(side=tk.LEFT, padx=10)
        label(frame, "教学视频").pack(side=tk.LEFT, padx=10)
        label(frame, "常见问题").pack(side=tk.LEFT, padx=10)

        return frame

    def main_left(self, parent):
        selectFramesList = R.widgets.selectFramesList(parent, lambda x: print(x),
                                                      labelTextList=["  模型中心", "选择模型", "训练模型", "校验模型", "预测结果", "  数据中心",
                                                                     "数据集管理"], pady=0)
        return selectFramesList

    def trainfile(self, filepath1):
        self.trainfile = filedialog.askopenfilename()
        filepath1.insert(0, self.trainfile)

    def testfile(self, filepath2):
        self.testfile = filedialog.askopenfilename()
        filepath2.insert(0, self.testfile)


if __name__ == '__main__':
    view = Viewer()
    view.run()
