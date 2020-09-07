import matplotlib.pyplot as plt
from Genetic_Vehicle_Plan.src.ui.button_acts import *


# noinspection PyAttributeOutsideInit
class Ui_Form(QtWidgets.QDialog, Ui_Form):
    def __init__(self):
        super(Ui_Form, self).__init__()
        self.setupUi(self)
        self.num_genes = []
        self.site_input = []
        self.vehicle_input = []
        self.err = []
        self.solution = []
        self.solution_fitness = []
        self.average_vehicle_cubic = float

        self.create_site_table_header()
        self.create_vehicle_table_header()

        # LineEdit
        font = QtGui.QFont()
        font.setPointSize(15)
        self.lineEdit_1.setText('150')
        self.lineEdit_1.setFont(font)
        self.lineEdit_2.setText('0.6')
        self.lineEdit_2.setFont(font)
        self.lineEdit_3.setText('0.001')
        self.lineEdit_3.setFont(font)
        self.lineEdit_5.setText('5')
        self.lineEdit_5.setFont(font)
        self.lineEdit_6.setText('20')
        self.lineEdit_6.setFont(font)

        # Add combobox
        self.comboBox.addItems(SELECTION)
        self.comboBox.setCurrentIndex(5)
        self.comboBox_2.addItems(CROSSOVER)
        self.comboBox_2.setCurrentIndex(2)
        self.comboBox_3.addItems(MUTATION)
        self.comboBox_3.setCurrentIndex(1)
        self.comboBox.setFont(font)
        self.comboBox_2.setFont(font)
        self.comboBox_3.setFont(font)

        # Click action
        self.toolButton.clicked.connect(lambda: self.button_load())
        self.toolButton_4.clicked.connect(lambda: self.run())

    def button_load(self):
        self.site_data = []
        self.vehicle_data = []
        self.thread = button_load()
        self.thread.signal.connect(self.callback)
        self.thread.start()

    def callback(self, msg):
        [self.site_data, self.vehicle_data] = msg
        self.create_site_table()
        self.create_vehicle_table()

    def button_run(self):
        # Input vehicle and site data from table
        self.load_vehicle_data()
        self.load_site_data()
        self.thread = button_run(vehicle_data=self.vehicle_input,
                                 site_data=self.site_input,
                                 num_generations=int(self.lineEdit_5.text()),
                                 num_parents_mating=int(self.lineEdit_6.text()),
                                 num_genes=self.num_genes,
                                 sol_per_pop=int(self.lineEdit_1.text()),
                                 parent_selection_type=self.comboBox.currentText(),
                                 crossover_type=self.comboBox_2.currentText(),
                                 crossover_probability=float(self.lineEdit_2.text()),
                                 mutation_type=self.comboBox_3.currentText(),
                                 mutation_probability=float(self.lineEdit_3.text()),
                                 verbose=False,
                                 multi_processing=True,
                                 process_bar=[self.pbar, self.label_15])
        self.thread.signal.connect(self.callback_run)
        self.thread.start()

        positives = [y for x in self.err for y in x if y >= 0]
        negatives = [y for x in self.err for y in x if y < 0]
        print(self.solution)
        print(self.solution_fitness)
        print("The total delay on site: {delay}, total lost for camp: {lost}".format(delay=sum(positives),
                                                                                     lost=sum(negatives)))
        # After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
        self.plot_result()

    def callback_run(self, msg):
        [self.solution,
         self.solution_fitness,
         self.solution_idx,
         self.err,
         self.current_site_data,
         self.current_vehicle_state,
         self.best_solutions_fitness] = msg

    def create_site_table_header(self):
        # ======================给site table设置行列表头============================
        font = QtGui.QFont()
        font.setPointSize(10)
        self.tableWidget.setColumnCount(len(SITE_HEADER))
        self.tableWidget.setHorizontalHeaderLabels(SITE_HEADER)
        self.tableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tableWidget.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeToContents)
        headitem = self.tableWidget.horizontalHeader()
        headitem.setFont(font)

    def create_vehicle_table_header(self):
        # ======================给vehicle table设置行列表头============================
        font = QtGui.QFont()
        font.setPointSize(10)
        self.tableWidget_2.setColumnCount(len(VEHICLE_HEADER))
        self.tableWidget_2.setHorizontalHeaderLabels(VEHICLE_HEADER)
        self.tableWidget_2.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tableWidget_2.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tableWidget_2.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tableWidget_2.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.tableWidget_2.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.tableWidget_2.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.tableWidget_2.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        headitem = self.tableWidget_2.horizontalHeader()
        headitem.setFont(font)

    def create_site_table(self):
        self.tableWidget.clear()
        self.create_site_table_header()
        # ===========读取表格，转换表格，===========================================
        self.tableWidget.setRowCount(len(self.site_data))
        font = QtGui.QFont()
        font.setPointSize(10)

        # ================遍历表格每个元素，同时添加到tablewidget中========================
        for i in range(len(self.site_data)):
            for j in range(len(SITE_HEADER)):
                newItem = QTableWidgetItem(self.site_data[i][j])
                newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.tableWidget.setItem(i, j, newItem)
                newItem.setFont(font)
        headitem = self.tableWidget.verticalHeader()
        headitem.setFont(font)

        self.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def create_vehicle_table(self):
        self.tableWidget_2.clear()
        self.create_vehicle_table_header()
        # ===========读取表格，转换表格，===========================================
        self.tableWidget_2.setRowCount(len(self.vehicle_data))
        font = QtGui.QFont()
        font.setPointSize(10)

        # ================遍历表格每个元素，同时添加到tablewidget中========================
        for i in range(len(self.vehicle_data)):
            for j in range(len(VEHICLE_HEADER)-1):
                newItem = QTableWidgetItem(self.vehicle_data[i][j])
                newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.tableWidget_2.setItem(i, j, newItem)
                newItem.setFont(font)
        headitem = self.tableWidget_2.verticalHeader()
        headitem.setFont(font)

        self.tableWidget_2.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def load_vehicle_data(self):
        self.vehicle_input = []
        for i in range(len(self.vehicle_data)):
            line = []
            for j in range(len(VEHICLE_HEADER)-1):
                line.append(self.tableWidget_2.item(i, j).text())
            _ = vehicle_state_input(line)
            self.vehicle_input.append(_)
        # The average vehicle cubic
        self.average_vehicle_cubic = numpy.average(numpy.asarray([int(x.cubic) for x in self.vehicle_input]))

    def load_site_data(self):
        self.site_input = []
        for i in range(len(self.site_data)):
            line = []
            for j in range(len(SITE_HEADER)):
                line.append(self.tableWidget.item(i, j).text())
            line.append(self.average_vehicle_cubic)
            _ = site_data_input(line)
            self.site_input.append(_)
        self.num_genes = numpy.asarray([int(x.n_deliver) for x in self.site_input])
        cubic_total = sum(numpy.asarray([int(x.demand_cubic) for x in self.site_input]))
        print('There are {num_car} vehicles available, the total planned cubic is {cubic_total}'
              .format(num_car=len(self.vehicle_data), cubic_total=cubic_total))

    def run(self):
        # Input vehicle and site data from table
        self.load_vehicle_data()
        self.load_site_data()

        # Run genetic algorithm
        ga_instance = GA(vehicle_data=self.vehicle_input,
                         site_data=self.site_input,
                         num_generations=int(self.lineEdit_5.text()),
                         num_parents_mating=int(self.lineEdit_6.text()),
                         num_genes=self.num_genes,
                         sol_per_pop=int(self.lineEdit_1.text()),
                         parent_selection_type=self.comboBox.currentText(),
                         crossover_type=self.comboBox_2.currentText(),
                         crossover_probability=float(self.lineEdit_2.text()),
                         mutation_type=self.comboBox_3.currentText(),
                         mutation_probability=float(self.lineEdit_3.text()),
                         verbose=False,
                         multi_processing=True,
                         process_bar=[self.pbar, self.label_15])

        print("Starting Genetic Algorithm")
        ga_instance.run()

        solution, \
        solution_fitness, \
        solution_idx, \
        err, \
        self.current_site_data, \
        self.current_vehicle_state, \
        self.best_solutions_fitness \
            = ga_instance.best_solution()

        positives = [y for x in err for y in x if y >= 0]
        negatives = [y for x in err for y in x if y < 0]
        self.label_16.setText("总损失时间：{solution_fitness}, 工地等待总时间: {delay}, 车辆等待总时间: {lost}".format(
            solution_fitness=int(solution_fitness),
            delay=sum(positives),
            lost=sum(negatives)))
        # After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
        self.plot_result()

    def plot_result(self, title="Iteration vs. Time Costs", xlabel="Generation", ylabel="Time Costs", linewidth=3):
        """
        Creates and shows a plot that summarizes how the fitness value evolved by generation. Can only be called after completing at least 1 generation.
        If no generation is completed, an exception is raised.
        """
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        x = range(len(self.best_solutions_fitness))
        y = 1 / numpy.asarray(self.best_solutions_fitness)
        ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        ax.plot(x, y)
        self.canvas.draw()

        font = QtGui.QFont()
        font.setPointSize(10)
        for i, vehicles in enumerate(self.current_vehicle_state):
            order_str = ""
            for order in vehicles.order_num:
                if order != 'None':
                    order_str += order + ';'
            newItem = QTableWidgetItem(order_str)
            newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.tableWidget_2.setItem(i, 6, newItem)
            newItem.setFont(font)
            pass


if __name__ == '__main__':
    import sys
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    # 创建QApplication类实例，用来获得参数
    app = QApplication(sys.argv)
    # 创建一个窗口
    MainWindow = Ui_Form()
    # 显示窗口
    MainWindow.show()
    # 进入程序的主循环、并通过exit函数确保主循环安全结束
    sys.exit(app.exec_())
