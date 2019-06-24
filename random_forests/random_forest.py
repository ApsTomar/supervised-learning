import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import data
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data.set_path(path)

iris_data, iris_target = make_classification(n_samples=120, n_features=4,
                                            n_informative=3, n_redundant=0,
                                            random_state=0, shuffle=False,
                                            n_classes=3)
clf = RandomForestClassifier(n_estimators=120, max_depth=2)
clf.fit(iris_data, iris_target)
test_data = data.load_test_data()
test_labels = data.load_test_labels()
prediction = clf.predict(test_data)
print accuracy_score(prediction, test_labels)
