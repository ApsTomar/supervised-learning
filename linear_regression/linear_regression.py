import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn import linear_model
from dataset import data
from sklearn.metrics import explained_variance_score

data.set_path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
X_train, X_test, y_train, y_test = data.load_house_data()
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
print('r-squared score on training data: %f' % (reg.score(X_train, y_train)))
print('r-squared score on testing data: %f' % (reg.score(X_test, y_test)))
print('explained_variance_score: %f' % explained_variance_score(pred, y_test))
