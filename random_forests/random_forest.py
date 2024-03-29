import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import data
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, explained_variance_score

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data.set_path(path)

# Random Forest Classifier:
iris_data, iris_target = data.load_iris_data()
clf = RandomForestClassifier(n_estimators=40, max_depth=2)
clf.fit(iris_data, iris_target)
scores = cross_val_score(clf, iris_data, iris_target, cv=3, scoring='accuracy')
print("Random Forest Classifier: cross_validation_score: %f" % scores.mean())
test_data = data.load_test_data()
test_labels = data.load_test_labels()
prediction = clf.predict(test_data)
print("Random Forest Classifier: accuracy score : %f" % accuracy_score(prediction, test_labels))

# Random Forest Regressor:
X_train, X_test, y_train, y_test = data.load_house_data()
reg = RandomForestRegressor(n_estimators=100, random_state=0)
reg.fit(X_train, y_train)
reg_pred = reg.predict(X_test)
print('Random Forest Regressor: r-squared score: %f' % reg.score(X_test, y_test))
print('Random Forest Regressor: explained_variance_score: %f' % explained_variance_score(reg_pred, y_test))
