# Maze-PyQt5
Use VC++/Python to solve the Maze problem.
利用 VC++/Python 语言实现迷宫路径求解，要求图形界面，可用
递归或非递归方式。
## 任务分析
### 实现思路
1. 将迷宫单元用二维列表表示，-1 为墙，0 为可以走的路
2. 首先初始化格子，等待输入，当点击按下后接受输入的迷宫列数，行数
3. 生成迷宫矩阵，据此画出迷宫。
4. 点击走迷宫按钮后，根据全局变量矩阵 maz 由三种算法计算路径。
### 涉及的知识点
• 迷宫生成算法:dfs 算法,bfs 算法,Prim 算法,Kruskal 算法
• 走迷宫算法:dfs 算法,bfs 算法,A*算法,(迪杰斯特拉算法,弗洛伊德算法）
• PyQt5 的布局与作图
• bfs 如何走迷宫并打印出迷宫路径（与 dfs 记录路径不一样）
### 出现的问题
• 关于 Python 参数传递问题（值传递 or 址传递）
• 如何动态刷新
• 关于 pyqt5 函数的一些参数问题
![image](https://user-images.githubusercontent.com/72803316/126328914-06689520-e205-40d9-8bb1-ab55df5f2f98.png)
