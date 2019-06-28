import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import data, preprocessing
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data.set_path(path)

# AdaBoost Classifier:
iris_data, iris_target = make_classification(n_samples=10, n_features=4,
                                             n_informative=3, n_redundant=0,
                                             random_state=0, shuffle=False)

decisionTreeClf = DecisionTreeClassifier(criterion='gini', max_depth=1)
decisionTreeClf.fit(iris_data, iris_target)
dt_scores = cross_val_score(decisionTreeClf, iris_data, iris_target, cv=5, scoring='accuracy')
print("Decision tree accuracy: %f" % dt_scores.mean())
adaBoostClf = AdaBoostClassifier(base_estimator=decisionTreeClf, n_estimators=10, random_state=0)
adaBoostClf.fit(iris_data, iris_target)
ab_scores = cross_val_score(adaBoostClf, iris_data, iris_target, cv=3, scoring='accuracy')
print("AdaBoost accuracy: %f" % ab_scores.mean())

# AdaBoost Regressor:
preprocessing.set_path(path)
# set vis = True for description of dataset and data visualization
X_train, X_test, y_train, y_test = preprocessing.load_house_data(vis=False)
reg = AdaBoostRegressor(n_estimators=50, learning_rate=0.1, loss='exponential')
reg.fit(X_train, y_train)
reg_pred = reg.predict(X_test)
print('AdaBoost Regressor: r-squared score: %f' % reg.score(X_train, y_train))
print('AdaBoost Regressor: explained_variance_score: %f' % explained_variance_score(reg_pred, y_test))
