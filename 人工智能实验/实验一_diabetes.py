from sklearn import datasets
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

datasets = datasets.load_diabetes()
print(datasets)
# separate the data into train_test
x_train = datasets.data[:-20]
y_train = datasets.target[:-20]
x_test = datasets.data[-20:]
y_test = datasets.target[-20:]
# train the model
line = linear_model.LinearRegression()
line.fit(x_train, y_train)
# predict
print("x_test is:")
print(x_test)
print("Predicted result is:")
print(line.predict(x_test))
print("The actual result is:")
print(y_test)
# predict in different elements
plt.figure(figsize=(8, 12))
for f in range(0, 10):
    # separate the data into train_test
    xi_test = x_test[:, f]
    xi_train = x_train[:, f]
    xi_test = xi_test[:, np.newaxis]
    xi_train = xi_train[:, np.newaxis]
    # train the model
    linreg = linear_model.LinearRegression()
    linreg.fit(xi_train, y_train)
    # predict
    y = linreg.predict(xi_test)
    # draw
    plt.subplot(5, 2, f+1)
    plt.scatter(xi_test, y_test, color='k')
    plt.plot(xi_test, y, color='b', linewidth=3)
plt.show()


