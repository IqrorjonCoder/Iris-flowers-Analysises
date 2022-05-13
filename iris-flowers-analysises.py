import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


def load_data(file_url):
    return pd.read_csv(file_url)


def summurize_dataset(data):
    # --------------   dataset   shape   --------------
    print(data.shape)

    # --------------   first  10  lines  --------------
    print(data.head(10))

    # -------------- statistical summary --------------
    print(data.describe())

    # -------------- class distribution  --------------
    print(data.groupby('class').size())


def print_plot_univariate(data_set):
    data_set.hist()
    plt.show()


def print_plot_multivariate(data_set):
    pd.plotting.scatter_matrix(data_set, color='#edc404', hist_kwds={'color': '#edc404'})
    plt.show()


def my_print_and_test_models(data):
    X = data.values[:, :4]
    y = data.values[:, 4]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=1)

    # DecisionTree
    model = DecisionTreeClassifier()
    kfold = KFold(n_splits=4)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    print('DecisionTree: %f (%f)' % (cv_results.mean(), cv_results.std()))

    # GaussianNB
    model = GaussianNB()
    kfold = KFold(n_splits=4)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    print('GaussianNB: %f (%f)' % (cv_results.mean(), cv_results.std()))

    # KNeighbors
    model = KNeighborsClassifier()
    kfold = KFold(n_splits=4)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    print('KNeighbors: %f (%f)' % (cv_results.mean(), cv_results.std()))

    # LogisticRegression
    model = LogisticRegression(solver='liblinear', multi_class='ovr')
    kfold = KFold(n_splits=4)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    print('LogisticRegression: %f (%f)' % (cv_results.mean(), cv_results.std()))

    # LinearDiscriminant
    model = LinearDiscriminantAnalysis()
    kfold = KFold(n_splits=4)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    print('LinearDiscriminant: %f (%f)' % (cv_results.mean(), cv_results.std()))

    # SVM
    model = SVC(gamma='auto')
    kfold = KFold(n_splits=4)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    print('SVM: %f (%f)' % (cv_results.mean(), cv_results.std()))


def main():
    data = load_data("iris.csv")

    # summurize_dataset(data)

    # print_plot_univariate(data)

    # print_plot_multivariate(data)

    my_print_and_test_models(data)


main()
