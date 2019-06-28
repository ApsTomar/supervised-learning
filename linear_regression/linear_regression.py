import os, sys
from sklearn import linear_model
from dataset import preprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

preprocessing.set_path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# set vis = True for description of dataset and data visualization
X_train, X_test, y_train, y_test = preprocessing.load_house_data(vis=False)
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
print('r-squared score on training data: %f' % (reg.score(X_train, y_train)))
print('r-squared score on testing data: %f' % (reg.score(X_test, y_test)))
