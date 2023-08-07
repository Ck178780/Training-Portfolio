#Cluster analysis is a technique or method for combining data points into groups or clusters based
# on the numeric properties of the data points or the groups/clusters
# 
# Clustering or Cluster analysis is one of the main data-driven methods for creating new knowledge
# from data
# 
# The concept of "cluster" is not defined
# 
# Clusters are consequently defined by the algorithm that assigns a data point to a particular
# cluster or creates a clusters from a collection of data points.
# 
# CA is unsupervised learning
# Withiin machine learning and artificial intelligence, CA is one of the main methods to make
# Machines "self-orgainze", "self-solve" or "create new knowledge" from data
# Within Data Science, Cluster Analysis is considered an explorative data analysis method
# 
# Most cluster analysis algorithms are iterative algorithms
# 
# Cluster Analysis solutions depend on the model, the data, data structures, administrative rules,
# or the subjective will of the operator
# 
# CA may be automated
# 
# Algoritms may use "rules of thumb", "trial and failure", or "requirements" for particular desired 
# properties.
# 
# Cluster analysts be they human or machines, may have or not have prior opinions on cluster
# structures or what clusters should be found
# 
# These prior opinions may or may not be included in a CA
# 
# Achieved results from a cluster analysis may sometimes be hard to explain w/o any subject 
# matter knowledge or need to be speculative - this is why cluster analysis is regarded as 
# explorative.
# 
# Ca has a large number of applications. 
# 
# In CA the objective is not to predict a target class variable or to classify a data point
# to a class (classification)
# Neither ios the objective to predict values for a variable y(prediction)
# CA finds existing groupings or structures in the data as well as algorithmically defined
# clusters among the data points.
# Ca can be used to create new classes for algorithms to classify new data. Similarly, new or
# through cluster analysis defined cluster membership labels may be used to predict future
# data values.#


#K-NEXT NEIGHBOUR CLASSIFIER
# K-NN Classifier computes class membership for data points
# 
# K-NN Classifier is not a cluster analysis algorithm but we'll take a look at this classifier
# as the method is simple and easy to use to understand clusters and classes
# 
# K-NN Classifier is an old and well-known algorithm that intuitively or "artificially intelligently"
# classifies or places data points into classes or clusters based on the data points' closeness
# and similarity to data points having a known class or cluster membership
# 
# The K in the algorithm is the number if closest neighbouring data points to the measured
# data points to the measured data point according to a distance measure.
# -K could be 1: the closest known data point determines class or the cluster membership
#   for the tested data point.
#  -K could be 2: the two closest known data point determines class or the cluster membership
#   for the tested data point.
#  -K could be 5: the five closest known data point determines class or the cluster membership
#   for the tested data point.
# -K could be N, any number, and the N closest known data point determines class or cluster
# membership for the tested data point#

#The K is dependent on distance measurement and also class/cluster dependent
# -this means that the K must be selected with consideration for distance measurements
# and the number of classes
# -for example 3 classes and a K of 3 may create soem confusion in some datasets if the 
# distance measurement is the lowest average distance and the three closest data points
# belong to different classes with the same distance from the considered data point#

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings("ignore")

iris = datasets.load_iris()
DataF = pd.DataFrame(data=iris.data, columns=iris.feature_names)
DataF["Species"] = iris.target
print(DataF.head(5))

sns.set_style("whitegrid")
Pairplot_graph = sns.pairplot(DataF, kind="scatter", hue="Species", palette = "bright")
plt.show()

#This shwo how the classifier works using the methodology with the "train_test_split" method
#We start by selecting the four data columns as data X and the class membership column as Y.
X = DataF.iloc[:, :-1].values
Y = DataF.iloc[:, 4].values
X_train, X_test,Y_train, Y_test = train_test_split(X, Y, test_size=0.40)

#We create our Kay-Nearest_Neighbor classifier with Kay set to five to begin
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

#We print the confusion matrixes and classification report to get a view on how well our Kay-naerest classifier works
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

X = DataF.iloc[:, :-1].values
Y = DataF.iloc[:, 4].values
X_train, X_test,Y_train, Y_test = train_test_split(X, Y, test_size=0.40)

Accuracy_data = pd.DataFrame(columns=['K', 'Trained_accuracy', 'Tested_accuracy'])

for k in range (1, 40):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    Y_train_pred = classifier.predict(X_train)
    Tr_accuracy = accuracy_score(Y_train, Y_train_pred)
    Te_accuracy = accuracy_score(Y_test, Y_pred)
    accuracy_values = pd.DataFrame.from_dict({'K': [k], 'Trained_accuracy': [Tr_accuracy], 'Tested_accuracy': [Te_accuracy]})
    Accuracy_data = pd.concat([Accuracy_data, accuracy_values], ignore_index=True)
    
    print(Accuracy_data.head(5))
    
    fig, ax = plt.subplots()
    sns.lineplot(x='K', y='Trained_accuracy', data=Accuracy_data, ax=ax, color='red')
    ax2 = ax.twinx()
    sns.lineplot(x='K', y='Tested_accuracy', data=Accuracy_data, ax=ax2, color='blue')
    plt.show()

