import os, sys
from dataset import data
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data.set_path(path)

iris_data, iris_target = data.load_iris_data()
clf = GaussianNB()
clf.fit(iris_data, iris_target)
scores = cross_val_score(clf, iris_data, iris_target, cv=3, scoring='accuracy')
print("cross_validation_score: %f" % scores.mean())
test_data = data.load_test_data()
test_labels = data.load_test_labels()
prediction = clf.predict(test_data)
print("accuracy score : %f" % accuracy_score(prediction, test_labels))
