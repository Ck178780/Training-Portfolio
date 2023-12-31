#Decision Trees (DTs) are viewed as a non-parametric supervised learning method that can be used for
# classification and regression
# 
# DTs create a tree hierarchy of simple if-then-else decision rules from the dataset.
# 
# These rules divide the dataset into a partition which can be used to predict future values,
# for example in a standard Regression environment.
# 
# Decision tree regressors is a machine learning tool considered inflexible in terms of the
# "if-then-else" rules but extremely flexible in adapting to deviating data points and 
# therefore apt to overfit data and in need of regularization.
#
# 
# Some advantages:
# Simple to explain and iterpret
# -Rules generated by DTs may be useful to subject-matter experts, although in the case of 
# regression, the task of prediction is often the primary interest
# 
# DTs are a multi-output capable method
# -in some machine learning tasks, this property is of great interest, and some subject matter
# experts may want to expllore or confirm theories with this kind of property. When predicting values
# with regression, this property is of less interest unless the result is the best-performing model
# 
# There are statistical tests developed creating possibilities to validate a model using statistical tests
# and create scientifically proven statements about the reliability of the model
# 
# Decision trees can be visualized by tree graphs
# 
# 
# Some issues when training DTs Regression models that we'll learn to handle are
# 
# Dts models are apt to overfit data because of their flexibility. 
# -This is easy to handle by limiting the data points required to create a leaf of limiting
# the maximum depth of a tree. A created tree may also be "pruned" - have diverging leaves
# or branches removed from the tree
# 
# DTs are apt to adapt to data points because of their propensity to grow new branches and 
# leaves. This can be handled by training many competing models and choosing among the better
# trees
# 
# predictions of Dts are not continous, bit stepwise and constant.
# -This may create math issues when predicting continous variables and create extrapolation
# issues when predicting non-stationary variables.
# 
# DTs are computationally expensive and considered Np-complete.
# DTs algorithms often use computational shortcuts that don't ensure the finding of an optimal
# decision tree. This issue of the risk of sub-optimality can be handled by training many competing
# models and choosing among the better trees.
# 
# Dts need balanced datasets to function well.
# This means that datasets need to be balanced before training DTs or subdived into data
# with suitable qualities, for example, if the data resembles the logic problems such 
# as the XOR, parity, or similar constructions.#

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor

#Download the medical costs datast from Github an save it to your harddrive's working directory.
#This code is needed after the dataset has been downloaded.

#New_Dataset = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
#print(New_Dataset.head(10))
#New_Dataset.to_csv('insurance_data.csv')

#Load the dataset from local storage. Select the first column to be index column.
DataF = pd.read_csv('insurance_data.csv', index_col=0)
print(DataF.head(5))

#We classify the body mass index values into medical classes.
bmi_labels = ['Underweight', 'Healthy weight', 'Overweight', 'Obesity']
cut_bins = [0, 18.500, 24.900, 29.900, 150.000]
DataF['bmi_label'] = pd.cut(DataF['bmi'], bins=cut_bins, labels=bmi_labels)

#Plot the age vs. charges data and show the impact of the artificial bmi_label
rel_plot = sns.relplot(data=DataF, x="age", y="charges", hue="bmi_label", kind="scatter", facet_kws=dict(legend_out=False))
rel_plot.fig.suptitle('Scatterplot of medical charges vs. age', fontsize='x-large')
rel_plot.set_axis_labels("Age", "Medical charges", fontsize='large' )
rel_plot._legend.set_title("BMI class",)
plt.show()

#Select a smaller subset to model with our polynomial regression model and which we will use Ridge and Lasso Regression methods
DF_obese_smokers = DataF.loc[(DataF['sex'] == 'male') & (DataF['bmi_label'] == 'Obesity') & (DataF['smoker'] == 'yes')]
print(DF_obese_smokers.head(5))

#PLot the age vs. charges data for obese male smokers
rel_plot = sns.relplot(data=DF_obese_smokers, x="age", y="charges", hue="bmi_label", kind="scatter", facet_kws=dict(legend_out=False))
rel_plot.fig.suptitle('Scatterplot of medical charges vs. age for male overweight smokers', fontsize='x-large')
rel_plot.set_axis_labels("Age", "Medical charges", fontsize='large' )
rel_plot._legend.set_title("BMI class",)
plt.show()

#We create the dataframes we will use for the analysis and the dataframes that we will use for vizualization of the fit models.
X = DF_obese_smokers.iloc[:, 0].values.reshape(-1, 1)
Y = DF_obese_smokers.iloc[:, 6].values.reshape(-1, 1)
age_list = [g/100 for g in range(DF_obese_smokers['age'].min()*100, DF_obese_smokers['age'].max()*100)]
DF_plot_1 = pd.DataFrame(columns=['age'])
DF_plot_1['age'] = age_list
DF_plot_2 = DF_plot_1.copy(deep=True)

#We create teh models and fit them to the X and Y dataframes.
DT_regr2 = DecisionTreeRegressor(max_depth=2, min_samples_leaf=3)
DT_regr4 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=3)
DT_regr6 = DecisionTreeRegressor(max_depth=6, min_samples_leaf=3)
DT_regr8 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=3)

DT_regr2.fit(X, Y)
DT_regr4.fit(X, Y)
DT_regr6.fit(X, Y)
DT_regr8.fit(X, Y)

#We add model predictions to the visualization dataframe
DF_plot_1['DT_regr2'] = DT_regr2.predict(DF_plot_2)
DF_plot_1['DT_regr4'] = DT_regr4.predict(DF_plot_2)
DF_plot_1['DT_regr6'] = DT_regr6.predict(DF_plot_2)
DF_plot_1['DT_regr8'] = DT_regr8.predict(DF_plot_2)
print(DF_plot_1.head())

#WE graph the models' output predictions to compare the decision tree regression models with the data
fig, ax = plt.subplots()
sns.scatterplot(x=DF_obese_smokers.age, y=DF_obese_smokers.charges, color='black', ax=ax)
sns.lineplot(x=DF_plot_1.age, y=DF_plot_1.DT_regr2, color='red', ax=ax)
sns.lineplot(x=DF_plot_1.age, y=DF_plot_1.DT_regr4, color ='blue', ax=ax)
sns.lineplot(x=DF_plot_1.age, y=DF_plot_1.DT_regr6, color='green', ax=ax)
sns.lineplot(x=DF_plot_1.age, y=DF_plot_1.DT_regr8, color='yellow', ax=ax)

ax.legend(['DT_regr2', 'DT_regr4', 'DT_regr6', 'DT_regr8'])
plt.title('Data vs. Model predictions', fontsize=20)
plt.xlabel('age', fontsize=16)
plt.ylabel('charges', fontsize=16)
plt.show()

#We will test min_samples regualarisation on a Decision tree regressor model with a depth 4.
#We'll create the models and fit them to the X and Y dataframes
DT_regr4_ms3 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=3)
DT_regr4_ms5 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5)
DT_regr4_ms7 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=7)
DT_regr4_ms9 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=9)

DT_regr4_ms3.fit(X, Y)
DT_regr4_ms5.fit(X, Y)
DT_regr4_ms7.fit(X, Y)
DT_regr4_ms9.fit(X, Y)

#We add model predictions to the visualization dataframe
DF_plot_1['DT_regr4_ms3'] = DT_regr4_ms3.predict(DF_plot_2)
DF_plot_1['DT_regr4_ms5'] = DT_regr4_ms5.predict(DF_plot_2)
DF_plot_1['DT_regr4_ms7'] = DT_regr4_ms7.predict(DF_plot_2)
DF_plot_1['DT_regr4_ms9'] = DT_regr4_ms9.predict(DF_plot_2)
print(DF_plot_1.head())

#WE graph the models' output predictions to compare the decision tree regression models with the data
fig, ax = plt.subplots()
sns.scatterplot(x=DF_obese_smokers.age, y=DF_obese_smokers.charges, color='black', ax=ax)
sns.lineplot(x=DF_plot_1.age, y=DF_plot_1.DT_regr4_ms3, color='red', ax=ax)
sns.lineplot(x=DF_plot_1.age, y=DF_plot_1.DT_regr4_ms5, color ='blue', ax=ax)
sns.lineplot(x=DF_plot_1.age, y=DF_plot_1.DT_regr4_ms7, color='green', ax=ax)
sns.lineplot(x=DF_plot_1.age, y=DF_plot_1.DT_regr4_ms9, color='yellow', ax=ax)

plt.legend(['Data points', 'DT_regr4_ms3', 'DT_regr4_ms5', 'DT_regr4_ms7', 'DT_regr4_ms9'])
plt.title('Data vs. Model predictions', fontsize=20)
plt.xlabel('age', fontsize=16)
plt.ylabel('charges', fontsize=16)
plt.show()
