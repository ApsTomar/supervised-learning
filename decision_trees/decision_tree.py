import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import data, preprocessing
from sklearn import tree
from sklearn.metrics import accuracy_score, explained_variance_score
from sklearn.model_selection import cross_val_score

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data.set_path(path)

# Decision Tree Classifier:
iris_data, iris_target = data.load_iris_data()
clf = tree.DecisionTreeClassifier()
clf.fit(iris_data, iris_target)
scores = cross_val_score(clf, iris_data, iris_target, cv=3, scoring='accuracy')
print("Decision Tree Classifier: cross_validation_score: %f" % scores.mean())
test_data = data.load_test_data()
test_labels = data.load_test_labels()
prediction = clf.predict(test_data)
print("Decision Tree Classifier: accuracy score : %f" % accuracy_score(prediction, test_labels))

# Decision Tree Regressor:
preprocessing.set_path(path)
# set vis = True for description of dataset and data visualization
X_train, X_test, y_train, y_test = preprocessing.load_house_data(vis=False)
reg = tree.DecisionTreeRegressor()
reg.fit(X_train, y_train)
reg_pred = reg.predict(X_test)
print('Decision Tree Regressor: r-squared score: %f' % reg.score(X_test, y_test))
print('Decision Tree Regressor: explained_variance_score: %f' % explained_variance_score(reg_pred, y_test))
