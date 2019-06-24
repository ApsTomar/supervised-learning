import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data.set_path(path)

iris_data, iris_target = data.load_iris_data()
clf = RandomForestClassifier(n_estimators=120, max_depth=2)
clf.fit(iris_data, iris_target)
scores = cross_val_score(clf, iris_data, iris_target, cv=3, scoring='accuracy')
print("Random Forest accuracy: %f" %scores.mean())