import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns

path = ''


def set_path(p):
    global path
    path = p


# printing dataset related details:
data = pd.read_csv(path + 'dataset/house_data.csv')
training_data = data.copy()
print('Sample Dataset:\n')
print(data.head(5))
print('Dataset Descriptions:\n')
print('Dimensions of dataset:', data.shape)
print('Datatype of features:\n', data.dtypes)
print('Null data values in columns: %d' % (data.isnull().any().sum()), '/', len(data.columns))
print('Null data values in rows: %d' % (data.isnull().any(axis=1).sum()), '/', len(data))

features = data.iloc[:, 3:].columns.tolist()
target = data.iloc[:, 2].name

# finding correlation in data:
correlations = {}
for f in features:
    temp = data[[f, target]]
    feat = temp[f].values
    targ = temp[target].values
    corr_key = f + ' vs ' + target
    correlations[corr_key] = stats.pearsonr(feat, targ)[0]

correlated_data = pd.DataFrame(correlations, index=['Correlation']).T
correlated_data = correlated_data.loc[correlated_data['Correlation'].abs().sort_values(ascending=False).index]
print('Correlation of features with price:\n',correlated_data)

# scatter plot of sqft_living against price
x_axis = 'sqft_living'
y_axis = 'price'
data = pd.concat([training_data[y_axis], training_data[x_axis]], axis=1)
data.plot.scatter(x=x_axis, y=y_axis)
plt.show()

# violin plot of grade against price
x_axis = 'grade'
data = pd.concat([training_data[y_axis],training_data[x_axis]],axis=1)
violin_plot = sns.violinplot(x=x_axis,y=y_axis,data=data)
violin_plot.axis(ymin=0,ymax=2000000)
plt.show()

# box plot of no. of bedrooms against price
x_axis = 'bedrooms'
data = pd.concat([training_data[y_axis], training_data[x_axis]], axis=1)
box_plot = sns.boxplot(x=x_axis, y=y_axis, data=data)
box_plot.axis(ymin=0,ymax=2000000)
plt.show()

# box plot of no. of floors against price
x_axis = 'floors'
data = pd.concat([training_data[y_axis], training_data[x_axis]], axis=1)
box_plot = sns.boxplot(x=x_axis, y=y_axis, data=data)
box_plot.axis(ymin=0,ymax=2000000)
plt.show()

