# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from sklearn import linear_model  # 表示，可以调用sklearn中的linear_model模块进行线性回归。
import numpy as np

matplotlib.use("Qt5Agg")
cars = pd.read_excel('xxxx.xlsx')
ckg = cars[['chang', 'kuan', 'gao']]
c = ckg[:]
print(c.dtypes)
# c.chang = c.chang.convert_objects(convert_numeric=True, copy=False)
# c.kuan = c.kuan.convert_objects(convert_numeric=True, copy=False)
# c = c.apply(pd.to_numeric, errors="ignore") # ignore不能强转，结果还是object类型
c = c.apply(pd.to_numeric, errors="coerce")
print(c.dtypes)
# c.plot.scatter(x='chang', y='kuan')
# plt.show()

model = linear_model.LinearRegression()
np.isnan(c).any()
c.dropna(inplace=True)
np.isnan(c).any()
X = c['chang'].values.reshape(-1, 1)
y = c['kuan'].values.reshape(-1, 1)
model.fit(X, y)
print(model.intercept_)  # 截距
print(model.coef_)  # 线性模型的系数
a = model.predict([[4500]])
print(a[0][0])

X2 = [[2000], [4000], [6000], [9000]]
y2 = model.predict(X2)
c.plot.scatter(x='chang', y='kuan')
plt.plot(X2, y2, 'g-')
plt.show()

