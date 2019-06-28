import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn import linear_model
from dataset import preprocessing

preprocessing.set_path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# set vis = True for description of dataset and data visualization
X_train, X_test, y_train, y_test = preprocessing.load_house_data(vis=False)
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
print(reg.coef_,reg.intercept_)
# score of regression / udacity 
