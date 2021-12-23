from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot as plt
# generate 2 class dataset by ouselves
X, y = make_classification(n_samples=2000, n_classes=2, random_state=1)
 # split datasets into train/test sets (80%~20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
# generate a no model prediction
nm_probs = [0 for _ in range(len(y_test))]
# fit a model
model = RandomForestRegressor()
model.fit(X_train, y_train)
# predict probabilities
rf_probs = model.predict(X_test)
# calculate precision and recall for each threshold
nm_precision, nm_recall, _ = precision_recall_curve(y_test, nm_probs)
rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_probs)
# calculate scores
rf_pred = (rf_probs > 0.5)
nm_f1, nm_auc = f1_score(y_test, nm_probs), auc(nm_recall, nm_precision)
rf_f1, rf_auc = f1_score(y_test, rf_pred), auc(rf_recall, rf_precision)
# plot the roc curve for the model
plt.plot(nm_recall, nm_precision, linestyle='--', label='No model AUC = %0.2f'% nm_auc)
plt.plot(rf_recall, rf_precision, marker='.', label='Random forest AUC = %0.2f'% rf_auc)
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()