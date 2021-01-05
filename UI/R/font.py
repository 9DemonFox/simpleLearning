from tkinter.font import BOLD, NORMAL
from tkinter.font import Font as tkFont


class NormalWeiRuanYaHeiFont(tkFont):
    """ 微软雅黑1
    """

    def __init__(self, size):
        kwargs = {
            "family": "微软雅黑",
            "size": size,
            "weight": NORMAL
        }
        super().__init__(**kwargs)


class BoldWeiRuanYaHeiFont(tkFont):
    def __init__(self, size):
        kwargs = {
            "family": "微软雅黑",
            "size": size,
            "weight": BOLD
        }
        super().__init__(**kwargs)


if __name__ == '__main__':
    pass
