from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap



# load the datasets and separate the test and train data
datasets = datasets.load_iris()
x_train = datasets.data[:130, 0:2]
print(x_train)
print(x_train)
y_train = datasets.target[:130]
x_test = datasets.data[130:, 0:2]
y_test = datasets.target[130:]
standardScaler = StandardScaler()
standardScaler.fit(x_train)
x_train_stand = standardScaler.transform(x_train)


# draw the decision boundary
def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1)
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)


    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    plt.contourf(x0, x1, zz, cmap=custom_cmap)


# set the model and train the data
clf = SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(x_train, y_train)

x1_min, x1_max = x_train[:, 0].min(), x_train[:, 0].max()  # 第0列的范围
x2_min, x2_max = x_train[:, 1].min(), x_train[:, 1].max()  # 第1列的范围
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
grid_hat = clf.predict(grid_test)
grid_hat = grid_hat.reshape(x1.shape)
cm_light = ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = ListedColormap(['g', 'r', 'b'])
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolors='k', s=50, cmap=cm_dark)  # 样本
plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
plt.xlabel(u'length', fontsize=13)
plt.ylabel(u'width', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'SVM', fontsize=15)
plt.show()


# get the accuracy, precision and recall
print('高斯核函数SVM的精度为:{}'.format(accuracy_score(y_test, clf.predict(x_test))))
precision, recall, _ = precision_recall_curve(y_test, clf.predict(x_test), pos_label=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.figure(1)
plt.plot(precision, recall)
# plt.show()

