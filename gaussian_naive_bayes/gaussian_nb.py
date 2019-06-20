import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import data
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data.set_path(path)

iris_data, iris_target = data.load_iris_data()
test_data = data.load_test_data()
test_labels = data.load_test_labels()
clf = GaussianNB()
clf.fit(iris_data, iris_target)
prediction = clf.predict(test_data)
print accuracy_score(prediction, test_labels)
