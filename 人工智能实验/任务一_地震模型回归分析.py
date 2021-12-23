from sklearn import linear_model
from sklearn import linear_model
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
# 一般有表头:
data_set = pd.read_table('./quake.dat', sep=',', comment='@', header=None)
x = list(zip(data_set[0], data_set[1], data_set[2]))
x = np.array(x)
y = data_set[3]
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=2000)
line = linear_model.LinearRegression()
line.fit(x_train, y_train)
y_pre = line.predict(x_test)


