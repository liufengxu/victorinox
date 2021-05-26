import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


iris = pd.read_csv('iris.csv')

flag = False
if flag:
    # 展示变量之间的两两关系，无颜色
    g = sns.PairGrid(iris)
    g.map_diag(plt.hist)  # 对角线上展示分布直方图
    g.map(plt.scatter)

flag = False
if flag:
    # 展示变量之间的两两关系，有颜色
    g = sns.PairGrid(iris, hue="species")
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter)
    g.add_legend()

flag = False
if flag:
    # 使用vars选取部分数据
    g = sns.PairGrid(iris, vars=["sepal_length", "sepal_width"], hue="species")
    g.map(plt.scatter)

flag = False
if flag:
    # pairplot 写法更简洁
    g = sns.pairplot(iris, hue="species", palette="Set2", diag_kind="kde", height=2.5)

flag = True
if flag:
    # 上下三角形展示不同的内容
    g = sns.PairGrid(iris)
    g.map_upper(plt.scatter)
    g.map_lower(sns.kdeplot, fill=True)
    g.map_diag(sns.kdeplot, lw=3, legend=False)

flag = False
if flag:
    # 散点图
    sns.jointplot(x="sepal_length", y="sepal_width", data=iris)
    # 六边图
    sns.jointplot(x="sepal_length", y="sepal_width", kind="hex", data=iris)
    # 核密度
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
    sns.jointplot(x="sepal_length", y="sepal_width", kind="kde", color="g", data=iris)

flag = False
if flag:
    # iris数据的黑白渐变效果
    sns.set(rc={'axes.facecolor': 'black', 'figure.facecolor': 'white', 'axes.grid': False})
    f, ax = plt.subplots(figsize=(6, 6))
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
    sns.kdeplot(x=iris.sepal_length, y=iris.sepal_width, cmap=cmap, n_levels=60, shade=True)

flag = False
if flag:
    # 随机数据的黑白渐变效果
    sns.set(rc={'axes.facecolor': 'black', 'figure.facecolor': 'white', 'axes.grid': False})
    mean, cov = [0, 1], [(1, .5), (.5, 1)]
    data = np.random.multivariate_normal(mean, cov, 200)
    df = pd.DataFrame(data, columns=["x", "y"])
    f, ax = plt.subplots(figsize=(6, 6))
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
    sns.kdeplot(x=df.x, y=df.y, cmap=cmap, n_levels=60, shade=True)

plt.show()

