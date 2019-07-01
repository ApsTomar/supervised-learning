import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from dataset import data, preprocessing
from sklearn.metrics import explained_variance_score

# Gradient Boost Regressor:
preprocessing.set_path(os.path.dirname(os.path.abspath(__file__)))
preprocessing.set_path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# set vis = True for description of dataset and data visualization
X_train, X_test, y_train, y_test = preprocessing.load_house_data(vis=False)
reg = GradientBoostingRegressor(n_estimators=400, loss='ls', max_depth=5, learning_rate=0.2, min_samples_split=2)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
print('Gradient Boost Regressor: r-squared score: %f' % reg.score(X_train, y_train))
print('Gradient Boost Regressor: explained_variance_score: %f' % explained_variance_score(pred, y_test))

# XGB Regressor:
xgb_reg = XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.2, objective='reg:squarederror')
xgb_reg.fit(X_train, y_train)
xgb_pred = xgb_reg.predict(X_test)
print('Extreme Gradient Boost Regressor: r-squared score: %f' % xgb_reg.score(X_train, y_train))
print('Extreme Gradient Boost Regressor: explained_variance_score: %f' % explained_variance_score(xgb_pred, y_test))
