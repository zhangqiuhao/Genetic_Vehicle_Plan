#from Genetic_Vehicle_Plan.vehicle_timetable import *

#print(add_time('23:41', 30))

'''

if __name__ == '__main__':

    import multiprocessing
    import datetime
    import numpy
    import timeit
    import threading


    def process(x):
        v = pow(x, 2)
        return v


    items = [x for x in range(10000000)]

    print('start multi')
    pool = multiprocessing.Pool(4)
    start_time = timeit.default_timer()
    results = pool.map(process, items)
    pool.close()
    pool.join()
    print(timeit.default_timer() - start_time)

    print('start single')
    start_time = timeit.default_timer()
    a=[]
    for i in items:
        a.append(process(i))
    print(timeit.default_timer() - start_time)


if __name__ == '__main__':
    import sys
    from form import Ui_Form
    from PyQt5.Qt import QWidget, QApplication, QTableWidgetItem
    import psycopg2


    class myform(QWidget, Ui_Form):
        def __init__(self):
            super().__init__()
            self.setupUi(self)

            self.btn1.clicked.connect(self.clear)
            self.btn2.clicked.connect(self.load)
            self.show()

        def clear(self):
            pass

        def load(self):
            conn = psycopg2.connect("dbname=test1_data user=jm password=123")
            cur = conn.cursor()
            cur.execute('select * from table1')
            rows = cur.fetchall()
            row = cur.rowcount  # 取得记录个数，用于设置表格的行数
            vol = len(rows[0])  # 取得字段数，用于设置表格的列数
            cur.close()
            conn.close()

            self.table.setRowCount(row)
            self.table.setColumnCount(vol)

            for i in range(row):
                for j in range(vol):
                    temp_data = rows[i][j]  # 临时记录，不能直接插入表格
                    data = QTableWidgetItem(str(temp_data))  # 转换后可插入表格
                    self.table.setItem(i, j, data)


    app = QApplication(sys.argv)
    w = myform()
    app.exec_()




#QAbstractButton -QPushButton的使用
from PyQt5.QtWidgets import  QPushButton,QVBoxLayout,QWidget,QApplication
from PyQt5.QtGui import QIcon,QPixmap

import sys

class WindowClass(QWidget):
    def __init__(self,parent=None):
        super(WindowClass, self).__init__(parent)
        self.btn_1=QPushButton("Btn_1")
        self.btn_2=QPushButton("Btn_2")
        self.btn_3=QPushButton("&DownLoad")#快捷建设置，ALT+大写首字母
        self.btn_4 = QPushButton("Btn_4")

        self.btn_1.setCheckable(True)#设置已经被点击
        self.btn_1.toggle()#切换按钮状态
        self.btn_1.clicked.connect(self.btnState)
        self.btn_1.clicked.connect(lambda :self.wichBtn(self.btn_1))

        #self.btn_2.setIcon(QIcon('./image/add_16px_1084515_easyicon.net.ico'))#按钮按钮
        self.btn_2.setIcon(QIcon(QPixmap('./image/baidu.png')))
        self.btn_2.setEnabled(False)#设置不可用状态
        self.btn_2.clicked.connect(lambda :self.wichBtn(self.btn_2))

        self.btn_3.setDefault(True)#设置该按钮式默认状态的
        self.btn_3.clicked.connect(lambda :self.wichBtn(self.btn_3))

        self.btn_4.clicked.connect(lambda :self.wichBtn(self.btn_4))

        self.resize(400,300)
        layout=QVBoxLayout()
        layout.addWidget(self.btn_1)
        layout.addWidget(self.btn_2)
        layout.addWidget(self.btn_3)
        layout.addWidget(self.btn_4)

        self.setLayout(layout)

    def btnState(self):
        if self.btn_1.isChecked():
            print("Btn_1被单击")
        else:
            print("Btn_1未被单击")
    def wichBtn(self,btn):
        print("点击的按钮是：" , btn.text())

if __name__=="__main__":
    app=QApplication(sys.argv)
    win=WindowClass()
    win.show()
    sys.exit(app.exec_())

'''

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57

# -*- coding: utf-8 -*-
'''
TODO:LQD
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC
from PyQt5.QtWidgets import QApplication, QPushButton, QMainWindow, QVBoxLayout, QWidget


class QtDraw(QMainWindow):
    flag_btn_start = True

    def __init__(self):
        super(QtDraw, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.resize(800, 600)
        self.setWindowTitle('PyQt5 Draw')

        # TODO:这里是结合的关键
        self.fig = plt.Figure()
        self.canvas = FC(self.fig)
        self.btn_start = QPushButton(self)
        self.btn_start.setText('draw')
        self.btn_start.clicked.connect(self.slot_btn_start)

        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.btn_start)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def slot_btn_start(self):
        try:
            ax = self.fig.add_subplot(111)
            x = np.linspace(0, 100, 100)
            y = np.random.random(100)
            ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
            ax.plot(x, y)
            self.canvas.draw()  # TODO:这里开始绘制
        except Exception as e:
            print(e)


def ui_main():
    app = QApplication(sys.argv)
    w = QtDraw()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    ui_main()