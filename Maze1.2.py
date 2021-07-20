# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
import sys,random
from PyQt5.QtWidgets import QApplication,QMainWindow,QGraphicsItem
from PyQt5.QtCore import Qt,QRectF
import numpy as np
from queue import Queue,PriorityQueue

WIDTH,HEIGHT=800,800 #Graphicsview的尺寸
COL_INTERVAL,ROW_INTERVAL = 3,3 #格子间的间距
COL_LEN,ROW_LEN = 20,20  #格子的长度
COL_NUM,ROW_NUM = 35,35  #格子的数量
FIND,GENERATE,SPEED=0,0,0 #生成方式，走迷宫方式，刷新速度
maz=np.ones((ROW_NUM,COL_NUM)) #迷宫矩阵
dx,dy=ROW_NUM - 2,COL_NUM - 1 #终点
record,ans,process= [],[],[] #记录答案

class DFSg(object):
    def __init__(self, width=11, height=11):
        # 迷宫最小长宽为5
        assert width >= 5 and height >= 5, "Length of width or height must be larger than 5."

        # 确保迷宫的长和宽均为奇数
        self.width = (width // 2) * 2 + 1
        self.height = (height // 2) * 2 + 1
        self.start = [1, 0]
        self.destination = [self.height - 2, self.width - 1]
        self.matrix = None

    def generate_matrix_dfs(self):
        # 地图初始化，并将出口和入口处的值设置为0
        self.matrix = -np.ones((self.height, self.width))
        self.matrix[self.start[0], self.start[1]] = 0
        self.matrix[self.destination[0], self.destination[1]] = 0

        visit_flag = [[0 for i in range(self.width)] for j in range(self.height)]

        def check(row, col, row_, col_):
            temp_sum = 0
            for d in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                temp_sum += self.matrix[row_ + d[0]][col_ + d[1]]
            return temp_sum <= -3

        def dfs(row, col):
            visit_flag[row][col] = 1
            self.matrix[row][col] = 0
            if row == self.start[0] and col == self.start[1] + 1:
                return

            directions = [[0, 2], [0, -2], [2, 0], [-2, 0]]
            random.shuffle(directions)
            for d in directions:
                row_, col_ = row + d[0], col + d[1]
                if row_ > 0 and row_ < self.height - 1 and col_ > 0 and col_ < self.width - 1 and visit_flag[row_][
                    col_] == 0 and check(row, col, row_, col_):
                    if row == row_:
                        visit_flag[row][min(col, col_) + 1] = 1
                        self.matrix[row][min(col, col_) + 1] = 0
                    else:
                        visit_flag[min(row, row_) + 1][col] = 1
                        self.matrix[min(row, row_) + 1][col] = 0
                    dfs(row_, col_)

        dfs(self.destination[0], self.destination[1] - 1)
        self.matrix[self.start[0], self.start[1] + 1] = 0
class PRIMg(object):
    def __init__(self, width=11, height=11):
        assert width >= 5 and height >= 5, "Length of width or height must be larger than 5."

        self.width = (width // 2) * 2 + 1
        self.height = (height // 2) * 2 + 1
        self.start = [1, 0]
        self.destination = [self.height - 2, self.width - 1]
        self.matrix = None
    # 虽然说是prim算法，但是我感觉更像随机广度优先算法
    def generate_matrix_prim(self):
        # 地图初始化，并将出口和入口处的值设置为0
        self.matrix = -np.ones((self.height, self.width))

        def check(row, col):
            temp_sum = 0
            for d in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                temp_sum += self.matrix[row + d[0]][col + d[1]]
            return temp_sum < -3

        queue = []
        row, col = (np.random.randint(1, self.height - 1) // 2) * 2 + 1, (
                    np.random.randint(1, self.width - 1) // 2) * 2 + 1
        queue.append((row, col, -1, -1))
        while len(queue) != 0:
            row, col, r_, c_ = queue.pop(np.random.randint(0, len(queue)))
            if check(row, col):
                self.matrix[row, col] = 0
                if r_ != -1 and row == r_:
                    self.matrix[row][min(col, c_) + 1] = 0
                elif r_ != -1 and col == c_:
                    self.matrix[min(row, r_) + 1][col] = 0
                for d in [[0, 2], [0, -2], [2, 0], [-2, 0]]:
                    row_, col_ = row + d[0], col + d[1]
                    if row_ > 0 and row_ < self.height - 1 and col_ > 0 and col_ < self.width - 1 and self.matrix[row_][
                        col_] == -1:
                        queue.append((row_, col_, row, col))

        self.matrix[self.start[0], self.start[1]] = 0
        self.matrix[self.destination[0], self.destination[1]] = 0
class UnionSet(object):
	def __init__(self, arr):
		self.parent = {pos: pos for pos in arr}
		self.count = len(arr)
	def find(self, root):
		if root == self.parent[root]:
			return root
		return self.find(self.parent[root])
	def union(self, root1, root2):
		self.parent[self.find(root1)] = self.find(root2)
class KRUSKALg(object):
	def __init__(self, width = 11, height = 11):
		assert width >= 5 and height >= 5, "Length of width or height must be larger than 5."

		self.width = (width // 2) * 2 + 1
		self.height = (height // 2) * 2 + 1
		self.start = [1, 0]
		self.destination = [self.height - 2, self.width - 1]
		self.matrix = None

	# 最小生成树算法-kruskal（选边法）思想生成迷宫地图，这种实现方法最复杂。
	def generate_matrix_kruskal(self):
		# 地图初始化，并将出口和入口处的值设置为0
		self.matrix = -np.ones((self.height, self.width))

		def check(row, col):
			ans, counter = [], 0
			for d in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
				row_, col_ = row + d[0], col + d[1]
				if row_ > 0 and row_ < self.height - 1 and col_ > 0 and col_ < self.width - 1 and self.matrix[row_, col_] == -1:
					ans.append([d[0] * 2, d[1] * 2])
					counter += 1
			if counter <= 1:
				return []
			return ans

		nodes = set()
		row = 1
		while row < self.height:
			col = 1
			while col < self.width:
				self.matrix[row, col] = 0
				nodes.add((row, col))
				col += 2
			row += 2

		unionset = UnionSet(nodes)
		while unionset.count > 1:
			row, col = nodes.pop()
			directions = check(row, col)
			if len(directions):
				random.shuffle(directions)
				for d in directions:
					row_, col_ = row + d[0], col + d[1]
					if unionset.find((row, col)) == unionset.find((row_, col_)):
						continue
					nodes.add((row, col))
					unionset.count -= 1
					unionset.union((row, col), (row_, col_))

					if row == row_:
						self.matrix[row][min(col, col_) + 1] = 0
					else:
						self.matrix[min(row, row_) + 1][col] = 0
					break

		self.matrix[self.start[0], self.start[1]] = 0
		self.matrix[self.destination[0], self.destination[1]] = 0
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1089, 850)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(870, 420, 141, 41))
        font = QtGui.QFont()
        font.setFamily("华文行楷")
        font.setPointSize(13)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(880, 210, 151, 51))
        font = QtGui.QFont()
        font.setFamily("华文行楷")
        font.setPointSize(20)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(900, 550, 111, 41))
        font = QtGui.QFont()
        font.setFamily("华文行楷")
        font.setPointSize(20)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(870, 720, 141, 41))
        font = QtGui.QFont()
        font.setFamily("华文行楷")
        font.setPointSize(13)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(820, 670, 245, 28))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_7 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(14)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_5.addWidget(self.label_7)
        self.comboBox_2 = QtWidgets.QComboBox(self.layoutWidget)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.horizontalLayout_5.addWidget(self.comboBox_2)
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(0, 0, 805, 805))
        self.graphicsView.setStyleSheet("")
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        self.graphicsView.setBackgroundBrush(brush)
        self.graphicsView.setObjectName("graphicsView")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(800, 40, 301, 151))
        font = QtGui.QFont()
        font.setFamily("Lucida Calligraphy")
        font.setPointSize(56)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(820, 260, 255, 31))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(15)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.widget1 = QtWidgets.QWidget(self.centralwidget)
        self.widget1.setGeometry(QtCore.QRect(820, 310, 255, 31))
        self.widget1.setObjectName("widget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.widget1)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(15)
        font.setBold(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.widget1)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout_2.addWidget(self.lineEdit_2)
        self.widget2 = QtWidgets.QWidget(self.centralwidget)
        self.widget2.setGeometry(QtCore.QRect(820, 370, 240, 28))
        self.widget2.setObjectName("widget2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.widget2)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.comboBox = QtWidgets.QComboBox(self.widget2)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout_3.addWidget(self.comboBox)
        self.widget3 = QtWidgets.QWidget(self.centralwidget)
        self.widget3.setGeometry(QtCore.QRect(820, 610, 241, 27))
        self.widget3.setObjectName("widget3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget3)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_6 = QtWidgets.QLabel(self.widget3)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(15)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_4.addWidget(self.label_6)
        self.horizontalSlider = QtWidgets.QSlider(self.widget3)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalLayout_4.addWidget(self.horizontalSlider)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1089, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Maze"))
        self.pushButton.setText(_translate("MainWindow", "一键生成迷宫"))
        self.label_4.setText(_translate("MainWindow", "迷宫生成"))
        self.label_5.setText(_translate("MainWindow", "走迷宫"))
        self.pushButton_2.setText(_translate("MainWindow", "开始迷宫"))
        self.label_7.setText(_translate("MainWindow", "走迷宫算法"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "DFS"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "BFS"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "A*"))
        self.label_8.setText(_translate("MainWindow", "Maze"))
        self.label.setText(_translate("MainWindow", "迷宫行数目"))
        self.label_2.setText(_translate("MainWindow", "迷宫列数目"))
        self.label_3.setText(_translate("MainWindow", "迷宫生成算法"))
        self.comboBox.setItemText(0, _translate("MainWindow", "深度优先"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Prim算法"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Kruskal算法"))
        self.label_6.setText(_translate("MainWindow", "速度"))
class Board(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Board, self).__init__()
        self.setupUi(self)
        self.graphicsView.scene=QtWidgets.QGraphicsScene(0,0,WIDTH,HEIGHT)
        maze=Maze()
        maze.setPos(0,0)
        self.graphicsView.scene.addItem(maze)
        self.graphicsView.setScene(self.graphicsView.scene)
        self.connecter()
        self.show()
    def connecter(self):
        self.pushButton.clicked.connect(self.draw)
        self.pushButton_2.clicked.connect(self.start)
    def start(self):
        global SPEED,FIND,ROUTE,ans,record
        ans=[]
        record=[]
        SPEED= int(self.horizontalSlider.value())
        FIND= int(self.comboBox_2.currentIndex())
        self.search()
        maze = Maze()
        maze.setPos(0, 0)
        self.update(maze)
    def draw(self):
        global COL_INTERVAL,ROW_INTERVAL,WIDTH,record
        global COL_NUM,ROW_NUM,COL_LEN,ROW_LEN,GENERATE,maz,ans
        ans=[]
        record=[]
        COL_NUM=int(self.lineEdit_2.text())
        ROW_NUM=int(self.lineEdit.text())
        if COL_NUM>=5 and ROW_NUM>=5:
            GENERATE = int(self.comboBox.currentIndex())
            self.updateParameter()
            maze = Maze()
            maze.setPos(0, 0)
            self.update(maze)
        else:
            print("长宽必须大于等于五")
    def generate(self):
        global maz
        gen=Gen()
        maz=np.ones((ROW_NUM+2,COL_NUM+2))
        if(GENERATE==0):
            gen.dfsg()
        if(GENERATE == 1):
            gen.primg()
        if (GENERATE == 2):
            gen.todog()
    def updateParameter(self):
        global COL_INTERVAL, ROW_INTERVAL, WIDTH,dx,dy
        global COL_NUM, ROW_NUM, COL_LEN, ROW_LEN
        self.generate()
        ROW_NUM,COL_NUM=maz.shape
        COL_INTERVAL = int(0.1 * WIDTH / (COL_NUM-1))
        ROW_INTERVAL = int(0.1 * HEIGHT / (ROW_NUM-1))
        COL_LEN = int(0.9 * WIDTH / COL_NUM)
        ROW_LEN = int(0.9 * HEIGHT / ROW_NUM)
        dx = ROW_NUM - 2
        dy = COL_NUM - 1
    def search(self):
        global FIND,record,ans
        if (FIND== 0):
            record.append((1,0))
            dfs(1,0)
            ans=list(ans)
        if (FIND == 1):
            bfs((1,0))
        if (FIND == 2):
            Astar()
    def update(self,maze):
        self.graphicsView.scene.addItem(maze)
        self.graphicsView.setScene(self.graphicsView.scene)
        self.show()
class Gen():
    def dfsg(self):
        global maz
        k=DFSg(ROW_NUM,COL_NUM)
        k.generate_matrix_dfs()
        maz=k.matrix
    def primg(self):
        global maz
        k =PRIMg(ROW_NUM, COL_NUM)
        k.generate_matrix_prim()
        maz = k.matrix

    def todog(self):
        global maz
        k = KRUSKALg(ROW_NUM, COL_NUM)
        k.generate_matrix_kruskal()
        maz = k.matrix

    def check(self,temp):
        pass

    def get_next(self,temp):
        global stack
        dir = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        np.random.shuffle(dir)
        (x,y)=temp
        for (dx,dy) in dir:
            if((dx+x)>0 and (dx+x)<ROW_NUM-1 and (dy+y)<COL_NUM-1 and (dy+y)>0):
                if((x+dx,y+dy) not in stack):
                    print(dx+x,dy+y)
                    return (dx+x,dy+y)
        return None
def Keep(temp):
    global ans,record
    ans=tuple(temp)
def Save(temp):
    global process, record
    process = tuple(temp)
def dfs(x,y):
    global dx, dy,record,w
    # Save(record)
    # pp = processPaint()
    # pp.setPos(0, 0)
    # w.update(pp)

    # time.sleep(0.1)
    if x == dx and y == dy:
        Keep(record)
        return
    for (kx, ky) in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
        if (check(kx + x, ky + y)):
            record.append((kx + x, ky + y))
            dfs(kx + x, ky + y)
            record.pop()
def bfs(t):
    global que,dx,dy,record
    lis={}
    visited=[(1,0)]
    que=Queue()
    que.put(t)
    while que:
        temp=que.get()
        if temp[0]==dx and temp[1]==dy:
            break
        for (kx, ky) in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
            x=kx + temp[0]
            y=ky + temp[1]
            if (x > 0 and y > 0 and x < ROW_NUM and y < COL_NUM and maz[x][y] == 0 and (x,y) not in visited):
                que.put((x, y))
                visited.append((x, y))
                lis[(x,y)]=(temp[0],temp[1])
                if (x==dx and y==dy):
                    break
    record.append((dx,dy))
    (x,y)=lis[(dx,dy)]
    record.append((x, y))
    while (x,y)!=(1,0):
        (x,y)=lis[x,y]
        record.append((x, y))
    Keep(record)
def Astar():
    start = (1, 0)
    final = (ROW_NUM - 2, COL_NUM - 1)
    front = PriorityQueue()
    front.put(start)
    father = {}
    father[start] = None
    sum_cost = {}
    sum_cost[start] = 0
    while front:
        current = front.get()
        if current == final:
            break
        for (dx, dy) in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
            x = current[0] + dx
            y = current[1] + dy
            if isOK((x, y)):
                cost = sum_cost[current] + calcuCost(current, (x, y))
                if (x, y) not in sum_cost or cost < sum_cost[(x, y)]:
                    sum_cost[(x, y)] = cost
                    priority = cost + heuristic(start, (x, y))
                    front.put((x, y), priority)
                    father[(x, y)] = current
                    if (x, y) == final:
                        break
    temp=final
    while temp:
        record.append(temp)
        temp=father[temp]
    Keep(record)
def check(x, y):
    global maz,record,ROW_NUM,COL_NUM
    if (x >= 0 and y >= 0 and x < ROW_NUM and y < COL_NUM and maz[x][y] == 0 and (x, y) not in record):
        return True
    return False
def heuristic(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])
def isOK(a):
    return (a[0]>0 and a[1]>0 and a[0]<ROW_NUM and a[1]<COL_NUM and maz[a[0]][a[1]]==0)
def calcuCost(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])
class Maze(QGraphicsItem):
    def __init__(self):
        super(Maze, self).__init__()
    def boundingRect(self):
        return QRectF(0, 0, 800, 800)
    def paint(self, painter, option, widget):
        global COL_INTERVAL, ROW_INTERVAL
        global WIDTH, COL_NUM, ROW_NUM
        global COL_LEN, ROW_LEN, maz,ROUTE
        for i in range(COL_NUM):
            for j in range(ROW_NUM):
                if(maz[i][j]!=0):
                   painter.setPen(Qt.green)
                   painter.setBrush(Qt.white)
                   painter.drawRect(i*(COL_LEN+COL_INTERVAL),j*(ROW_LEN+ROW_INTERVAL),COL_LEN,ROW_LEN)
                if((i,j) in ans):
                   painter.setPen(Qt.yellow)
                   painter.setBrush(Qt.red)
                   painter.drawEllipse(i * (COL_LEN + COL_INTERVAL)+COL_LEN/4, j * (ROW_LEN + ROW_INTERVAL)+ROW_LEN/4, COL_LEN/2, ROW_LEN/2)
class processPaint(QGraphicsItem):
    def __init__(self):
        super(processPaint, self).__init__()
    def boundingRect(self):
        return QRectF(0, 0, 800, 800)
    def paint(self, painter, option, widget):
        global COL_INTERVAL, ROW_INTERVAL
        global WIDTH, COL_NUM, ROW_NUM
        global COL_LEN, ROW_LEN, maz,process
        for i in range(COL_NUM):
            for j in range(ROW_NUM):
                if(maz[i][j]!=0):
                   painter.setPen(Qt.green)
                   painter.setBrush(Qt.white)
                   painter.drawRect(i*(COL_LEN+COL_INTERVAL),j*(ROW_LEN+ROW_INTERVAL),COL_LEN,ROW_LEN)
                if((i,j) in record):
                   painter.setPen(Qt.yellow)
                   painter.setBrush(Qt.red)
                   painter.drawEllipse(i * (COL_LEN + COL_INTERVAL)+COL_LEN/4, j * (ROW_LEN + ROW_INTERVAL)+ROW_LEN/4, COL_LEN/2, ROW_LEN/2)
def main():
    global w
    app = QApplication(sys.argv)
    w=Board()
    w.show()
    sys.exit(app.exec_())
main()