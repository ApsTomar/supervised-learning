import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import data
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data.set_path(path)

iris_data, iris_target = make_classification(n_samples=100, n_features=4,
                                            n_informative=3, n_redundant=0,
                                            random_state=0, shuffle=False)

decisionTreeClf = DecisionTreeClassifier(criterion='entropy', max_depth=2)
decisionTreeClf.fit(iris_data,iris_target)
dt_scores = cross_val_score(decisionTreeClf, iris_data, iris_target, cv=5, scoring='accuracy')
print("Decision tree accuracy: %f" %dt_scores.mean())
adaBoostClf = AdaBoostClassifier(base_estimator=decisionTreeClf, n_estimators=10, random_state=0)
adaBoostClf.fit(iris_data,iris_target)
ab_scores = cross_val_score(adaBoostClf, iris_data, iris_target, cv=3, scoring='accuracy')
print ("AdaBoost accuracy: %f" %ab_scores.mean())