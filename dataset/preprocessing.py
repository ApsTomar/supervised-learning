import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

path = ''


def set_path(p):
    global path
    path = p


def describe_data(df):
    print('Sample Dataset:\n')
    print(df.head(5))
    print('Dataset Descriptions:\n')
    print('Dimensions of dataset:', df.shape)
    print('Datatype of features:\n', df.dtypes)
    print('Null data values in columns: %d' % (df.isnull().any().sum()), '/', len(df.columns))
    print('Null data values in rows: %d' % (df.isnull().any(axis=1).sum()), '/', len(df))


# finding correlation in data:
def find_correlations(features, target, df):
    correlations = {}
    for f in features:
        temp = df[[f, target]]
        feat = temp[f].values
        targ = temp[target].values
        corr_key = f + ' vs ' + target
        correlations[corr_key] = stats.pearsonr(feat, targ)[0]

    correlated_data = pd.DataFrame(correlations, index=['Correlation']).T
    correlated_data = correlated_data.loc[correlated_data['Correlation'].abs().sort_values(ascending=False).index]
    return correlated_data


def visualize_data(df):
    # scatter plot of sqft_living against price
    x_axis = 'sqft_living'
    y_axis = 'price'
    data = pd.concat([df[y_axis], df[x_axis]], axis=1)
    data.plot.scatter(x=x_axis, y=y_axis)
    plt.show()

    # violin plot of grade against price
    x_axis = 'grade'
    data = pd.concat([df[y_axis], df[x_axis]], axis=1)
    violin_plot = sns.violinplot(x=x_axis, y=y_axis, data=data)
    violin_plot.axis(ymin=0, ymax=2000000)
    plt.show()

    # box plot of no. of bedrooms against price
    x_axis = 'bedrooms'
    data = pd.concat([df[y_axis], df[x_axis]], axis=1)
    box_plot = sns.boxplot(x=x_axis, y=y_axis, data=data)
    box_plot.axis(ymin=0, ymax=2000000)
    plt.show()

    # box plot of no. of floors against price
    x_axis = 'floors'
    data = pd.concat([df[y_axis], df[x_axis]], axis=1)
    box_plot = sns.boxplot(x=x_axis, y=y_axis, data=data)
    box_plot.axis(ymin=0, ymax=2000000)
    plt.show()


def load_house_data(vis):
    data = pd.read_csv(path + '/dataset/house_data.csv')
    if vis:
        preprocess_data(data)
    features_data = data[
        ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 'view', 'sqft_basement',
         'waterfront', 'yr_built', 'lat', 'bedrooms', 'long']]
    X_train, X_test, y_train, y_test = train_test_split(features_data.values, data.price.values, test_size=0.2)
    return X_train, X_test, y_train, y_test


def preprocess_data(data):
    describe_data(data)
    features = data.iloc[:, 3:].columns.tolist()
    target = data.iloc[:, 2].name
    print(find_correlations(features, target, data))
    visualize_data(data)
