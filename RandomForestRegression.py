# Random Forest is a supervised learning technique that can be used for both Prediction and Classification
# 
# random forest regression or random forest ensembles for prediction consists of Decision Tree
# regressors
# 
# the random forest is an ensemble method. this means that a Random Forest consists of many DTs
# forming a forest. Depending  on the choice of method either the best performing DT represents 
# the Forest, or a number of DTs in the Random Forest decide Predictions through a method of
# voting where the winning Prediction is the Random Forest's decision
# 
# The Random Forest Ensemble can be likened to a group of experts where either the best performing
# expert rules the group or the group votes on the Prediction task and that the majority or
# the other criteria rules the group. The voting mechanism in the ensemble is often argued to 
# lesson the risk of overfitting as a function of relying on more than a single expert or
# single decision tree.
# 
# The Scikit-Learn Random forest which we'll use in this video is a number n of decision trees
# created by random draws from the training set. The n  Dts uses a voting mechanism to decide 
# it's prediction, in the current implementation of scikit-learn the mean predicted regression
# targets of the trees in the forest.#

#The basic random forest algorithm as currently implemneted in Scikit-Learn 1.2.2
# 
# 1.You decide the number N of DTs in the Random Forest, for eaxmple 5, 50, 100, or 1000
# and all the other parameters required for the particular choice of algorithm
# 
# 2.The algorithm creates a new subsample from the training data for every of the N Decision
# Trees and trains the DTs. For every tree in the forest, fit the tree to its subsample
# of training data
# 
# 3.For every new data point in need of a prediction (the test data ro future data), every N
# of the N Decision Trees makes a prediction and the final selected Prediction is determined by 
# the mean predicted regression targets of trees in the forest.#

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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

#We select a knwon subset of high-charge medical customers, in this case male smokers from the data and plot charges vs.age
DF_overw_smokers = DataF.loc[(DataF['smoker'] == 'yes') & ((DataF['bmi_label'] == 'Overweight') | (DataF['bmi_label'] == 'Obesity'))]
print(DF_overw_smokers.head(5))

#Plot the age vs. charges data for overweight smokers
rel_plot = sns.relplot(data=DF_overw_smokers, x="age", y="charges", hue="bmi_label", kind="scatter", facet_kws=dict(legend_out=False))
rel_plot.fig.suptitle('Scatterplot of medical charges vs. age for male overweight smokers', fontsize='x-large')
rel_plot.set_axis_labels("Age", "Medical charges", fontsize='large' )
rel_plot._legend.set_title("BMI class",)
plt.show()

#We select the obese smokers subset
DF_obese_smokers = DataF.loc[(DataF['smoker'] == 'yes') & (DataF['bmi_label'] == 'Obesity')]
print(DF_obese_smokers.head(5))

#Plot the age vs. charges data for overweight smokers
rel_plot = sns.relplot(data=DF_obese_smokers, x="age", y="charges", hue="bmi_label", kind="scatter", facet_kws=dict(legend_out=False))
rel_plot.fig.suptitle('Scatterplot of medical charges vs. age for male overweight smokers', fontsize='x-large')
rel_plot.set_axis_labels("Age", "Medical charges", fontsize='large' )
rel_plot._legend.set_title("BMI class",)
plt.show()

#Create the dataframes we will use for the modeling and dataframes for visualization of the models
X = DF_obese_smokers[["age"]].values
Y = DF_obese_smokers.iloc[:, 6]

age_list = [g/100 for g in range(DF_obese_smokers['age'].min()*100, DF_obese_smokers['age'].max()*100)]
DF_plot_1 = pd.DataFrame(columns=['age'])
DF_plot_1['age'] = age_list
DF_plot_2 = DF_plot_1.copy(deep=True)

#Create the models and fit them to the X and Y dataframes.
RF_regr = RandomForestRegressor(n_estimators=1000, max_depth=2, min_samples_leaf=9)

DT_regr2 = DecisionTreeRegressor(max_depth=2, min_samples_leaf=3)
DT_regr4 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=3)
DT_regr6 = DecisionTreeRegressor(max_depth=6, min_samples_leaf=3)
DT_regr4_ms3 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=3)
DT_regr4_ms5 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5)
DT_regr4_ms7 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=7)

RF_regr.fit(X, Y)
DT_regr2.fit(X, Y)
DT_regr4.fit(X, Y)
DT_regr6.fit(X, Y)
DT_regr4_ms3.fit(X, Y)
DT_regr4_ms5.fit(X, Y)
DT_regr4_ms7.fit(X, Y)

# We add model predictions to the visualization dataframe
DF_plot_1['RF_regr']= RF_regr.predict(DF_plot_2)
DF_plot_1['DT_regr2'] = DT_regr2.predict(DF_plot_2)
DF_plot_1['DT_regr4'] = DT_regr4.predict(DF_plot_2)
DF_plot_1['DT_regr6'] = DT_regr6.predict(DF_plot_2)
DF_plot_1['DT_regr4_ms3'] = DT_regr4_ms3.predict(DF_plot_2)
DF_plot_1['DT_regr4_ms5'] = DT_regr4_ms5.predict(DF_plot_2)
DF_plot_1['DT_regr4_ms7'] = DT_regr4_ms7.predict(DF_plot_2)
print(DF_plot_1.head())

#Graph the model's output predictions to compare the Random Forest's prediction with the DT regression models and data
fig, ax = plt.subplots()
DF_obese_smokers.plot(kind="scatter", x="age", y="charges", ax=ax, label="Data point", color="black")
DF_plot_1.plot(kind="line", x="age", y="DT_regr2", ax=ax, label="DT_regr2", color="red")
DF_plot_1.plot(kind="line", x="age", y="DT_regr4", ax=ax, label="DT_regr4", color="grey")
DF_plot_1.plot(kind="line", x="age", y="DT_regr6", ax=ax, label="DT_regr6", color="yellow")
DF_plot_1.plot(kind="line", x="age", y="DT_regr4_ms3", ax=ax, label="DT_regr4_ms3", color="brown")
DF_plot_1.plot(kind="line", x="age", y="DT_regr4_ms5", ax=ax, label="DT_regr4_ms5", color="blue")
DF_plot_1.plot(kind="line", x="age", y="DT_regr4_ms7", ax=ax, label="DT_regr4_ms7", color="green")
DF_plot_1.plot(kind="line", x="age", y="RF_regr", ax=ax, label="RF_regr", color="black", linewidth=3)
plt.legend(fontsize=8)
ax.set_xlabel("Age", fontsize=16)
ax.set_ylabel("Charges", fontsize=16)
fig.suptitle("")
ax.set_title("Data vs. Model Predictions", fontsize=20)
plt.show()

#Demonstration of the Random Forest's Regression inbuilt functionality to calculate and select importances for/from the data
#Note that the encoded features are numerically consistent except for binary 0-1 variable and that the encoded numerical
#distances are irrelevant for any other evaluation than as a "different item/group". The calculated importances for some
#variables are not numerically meaningful but gives a weak indication about the variables importance(region_emc, bmi_labels_enc)
le1 = preprocessing.LabelEncoder()
le2 = preprocessing.LabelEncoder()
le3 = preprocessing.LabelEncoder()
le4 = preprocessing.LabelEncoder()
DataF["sex_enc"] = le1.fit_transform(DataF["sex"])
DataF["smoker_enc"] = le2.fit_transform(DataF["smoker"])
DataF["region_enc"] = le3.fit_transform(DataF["region"])
DataF["bmi_label_enc"] = le4.fit_transform(DataF["bmi_label"])
print(DataF.head())

#Create dataframes to use for modeling, calculating importances and visualizing the models importances.
DF2_overw_smo = DataF.loc[(DataF['smoker'] == 'yes') & ((DataF['bmi_label'] == 'Overweight') | (DataF['bmi_label'] == 'Obesity'))]
DF2_obese_smo = DataF.loc[(DataF['smoker'] == 'yes') & (DataF['bmi_label'] == 'Obesity')]

X2 = DataF[['age', 'bmi', 'children', 'sex_enc', 'smoker_enc', 'region_enc', 'bmi_label_enc']]
Y2 = DataF['charges']
X3 = DF2_overw_smo[['age', 'bmi', 'children', 'sex_enc', 'smoker_enc', 'region_enc', 'bmi_label_enc']]
Y3 = DF2_overw_smo[['charges']]
X4 = DF2_obese_smo[['age', 'bmi', 'children', 'sex_enc', 'smoker_enc', 'region_enc', 'bmi_label_enc']]
Y4 = DF2_obese_smo['charges']

#Create the models and fit them to the X and Y dataframes.
RF_regr2 = RandomForestRegressor(n_estimators=1000, max_depth=3, min_samples_leaf=3, random_state=11)
RF_regr2.fit(X2, Y2)
RF_regr3 = RandomForestRegressor(n_estimators=1000, max_depth=3, min_samples_leaf=3, random_state=11)
RF_regr3.fit(X3, Y3)
RF_regr4 = RandomForestRegressor(n_estimators=1000, max_depth=3, min_samples_leaf=3, random_state=11)
RF_regr4.fit(X4, Y4)

#Create and sort a DataFrame with feature names and importance scores
DF2_features = pd.DataFrame({'feature': RF_regr2.feature_names_in_, 'importance': RF_regr2.feature_importances_})
DF2_features_sorted = DF2_features.sort_values(by='importance', ascending=False)

#Create a Seaborn Barplot of the importance scores and features
plt.figure(figsize=(13,6))
BP2_graph = sns.barplot(data=DF2_features_sorted, x='importance', y='feature', palette="bright")
BP2_graph.set_title('Feature importance for the entire insurance dataset')
BP2_graph.set(xlabel=None)
BP2_graph.set(ylabel=None)
BP2_graph.set(xticks=[])
for value in BP2_graph.containers:
    BP2_graph.bar_label(value, padding=3)
plt.show()    

#Create and sort a DataFrame with feature names and importance scores
DF3_features = pd.DataFrame({'feature': RF_regr3.feature_names_in_, 'importance': RF_regr3.feature_importances_})
DF3_features_sorted = DF3_features.sort_values(by='importance', ascending=False)

#Create a Seaborn Barplot of the importance scores and features
plt.figure(figsize=(13,6))
BP3_graph = sns.barplot(data=DF3_features_sorted, x='importance', y='feature', palette="bright")
BP3_graph.set_title('Feature importance for overweight smokers')
BP3_graph.set(xlabel=None)
BP3_graph.set(ylabel=None)
BP3_graph.set(xticks=[])
for value in BP3_graph.containers:
    BP3_graph.bar_label(value, padding=3)
plt.show()  

#Create and sort a DataFrame with feature names and importance scores
DF4_features = pd.DataFrame({'feature': RF_regr4.feature_names_in_, 'importance': RF_regr4.feature_importances_})
DF4_features_sorted = DF4_features.sort_values(by='importance', ascending=False)

#Create a Seaborn Barplot of the importance scores and features
plt.figure(figsize=(13,6))
BP4_graph = sns.barplot(data=DF4_features_sorted, x='importance', y='feature', palette="bright")
BP4_graph.set_title('Feature importance for obese smokers')
BP4_graph.set(xlabel=None)
BP4_graph.set(ylabel=None)
BP4_graph.set(xticks=[])
for value in BP4_graph.containers:
    BP4_graph.bar_label(value, padding=3)
plt.show() 


