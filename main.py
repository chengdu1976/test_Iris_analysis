# import packages
import numpy as np 
import sklearn 
import matplotlib.pyplot as plt
%matplotlib inline


# load data
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
list(iris)

# store features and labels in X and y
X = iris.data
y = iris.target

# print the first 5 rows of X
X.head()

# Print data types (numerical or categorical) and check if there are any missing values
X.info()

# Generate descriptive statistics (note that the data has been standardized already)
X.describe()


# plot histograms of the features with matplotlib to see their distributions
X.hist(bins=25)
plt.show()

# Plot the most important feature pairs with a scatter plot
from pandas.plotting import scatter_matrix
scatter_matrix(X, figsize=(12, 10))
plt.show()


# we only use 2 features for training
X = iris.data[["petal length (cm)", "petal width (cm)"]]

# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)


# train a decision tree classifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3, random_state=13)
model.fit(X_train, y_train)


# plot the decision tree
from sklearn.tree import plot_tree
plot_tree(model, filled=True)


# make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# calculate accuracy on training and test set
from sklearn.metrics import accuracy_score
acc_train = accuracy_score(y_pred_train, y_train)
acc_test = accuracy_score(y_pred_test, y_test)
print(acc_train)
print(acc_test)

# plot several decision trees with different max_depth

def plot_decision_boundaries(X_train, y_train, max_depth):
    # train another decision tree classifier with variable max_depth for plotting
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=13)
    model.fit(X_train.values, y_train.values)

    # use subplots to plot the decision tree and the decision boundaries
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 5))

    # plot the decision tree
    plot_tree(model, filled=True, ax=ax1)
    ax1.set_title("Decision tree model (max_depth = {})".format(max_depth))

    # plot params
    xlim0, xlim1 = 0, 7
    ylim0, ylim1 = 0, 3

    # plot labels
    ax2.scatter(X_train.values[y_train.values == 0, 0], X_train.values[y_train.values == 0, 1], color='orange', marker='s', label="Iris setosa")
    ax2.scatter(X_train.values[y_train.values == 1, 0], X_train.values[y_train.values == 1, 1], color='green', marker='^', label="Iris versicolor")
    ax2.scatter(X_train.values[y_train.values == 2, 0], X_train.values[y_train.values == 2, 1], color='purple', marker='o', label="Iris virginica")
    ax2.set_xlabel("Petal length")
    ax2.set_ylabel("Petal width")
    ax2.legend(loc="upper left")
    ax2.set_title("Labels and decision boundaries (max_depth = {})".format(max_depth))

    # plot the predictions
    from matplotlib.colors import ListedColormap
    my_cmap = ListedColormap(['orange', 'green', 'purple'])
    xx, yy = np.meshgrid(np.linspace(xlim0, xlim1, 100), np.linspace(ylim0, ylim1, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, alpha=0.2, cmap=my_cmap)

    # plot the decision boundaries
    print("tree thresholds:", model.tree_.threshold)
    if max_depth == 1:
        a = model.tree_.threshold[[0]]
        ax2.plot([a, a], [ylim0, ylim1], "k--")
    elif max_depth == 2:
        a, b = model.tree_.threshold[[0, 2]]
        ax2.plot([a, a], [ylim0, ylim1], "k--")
        ax2.plot([a, xlim1], [b, b], "k--")
    elif max_depth == 3:
        a, b, c, d = model.tree_.threshold[[0, 2, 3, 6]]
        ax2.plot([a, a], [ylim0, ylim1], "k--")
        ax2.plot([a, xlim1], [b, b], "k--")
        ax2.plot([c, c], [ylim0, b], "k--")
        ax2.plot([d, d], [b, ylim1], "k--")


for i in range(1, 4):
    plot_decision_boundaries(X_train, y_train, i)

