#Regularization of regression equations is a method where the regression parameters are penalized
# to favour small parameter values or feature selection is penalized to favour smaller non-complex
# models. The primary intention of penalizing the equations is to avoid or lessen overfitting
# 
# Ridge and Lasso regression is a type of regularization of the regression equations with the main
# purpose of creating a lower-variance regression equation through the introduction of a small
# "bias-including term" to the equation.
# 
# The effect of the "bias term" is in practice to smoothen a regression "line curve" to avoid
# extensive overfitting or to "lower" a regression line to avoid "over-adapting" to extreme value 
# data. In some datasets where data is highly or multi-correlated, Ridge regression may lower the 
# the model variance accross samples of the regression coeffiecients.
# #

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

#Download the medical costs dataset from Github and save to storage
#This code is not needed once you have downloaded the dataset

new_dataset = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
print(new_dataset.head(10))
new_dataset.to_csv('insurance.csv')

#Load the dataset from local storage. Select the first column to be the index column
DataF = pd.read_csv('insurance.csv', index_col=0)
print(DataF.head())

#Classify the Body Mass Index variable into medical classes
bmi_labels = ['Underweight', 'Healthy weight', 'Overweight', 'Obesity']
cut_bins = [0, 18.500,24.900, 29.900, 150.000]
DataF['bmi_label'] = pd.cut(DataF['bmi'], bins=cut_bins, labels=bmi_labels)
print(DataF.head())

#We select a knwon subset of high-charge medical customers, in this case male smokers from the data and plot charges vs.age
DF_male_overw_smokers = DataF.loc[(DataF['sex'] == 'male') & (DataF['smoker'] == 'yes') & ((DataF['bmi_label'] == 'Overweight') | (DataF['bmi_label'] == 'Obesity'))]
print(DF_male_overw_smokers.head(5))
rel_plot = sns.relplot(data=DF_male_overw_smokers, x="age", y="charges", hue="bmi_label", kind="scatter", facet_kws=dict(legend_out=False))
rel_plot.fig.suptitle('Scatterplot of medical charges vs. age for male overweight smokers', fontsize='x-large')
rel_plot.set_axis_labels("Age", "Medical charges", fontsize='large' )
rel_plot._legend.set_title("BMI class",)
plt.show()

#Select a smaller subset to model with our polynomial regression model and which we will use Ridge and Lasso Regression methods
DF_male_obese_smokers = DataF.loc[(DataF['sex'] == 'male') & (DataF['smoker'] == 'yes') & (DataF['bmi_label'] == 'Obesity')]
print(DF_male_overw_smokers.head(5))
rel_plot = sns.relplot(data=DF_male_overw_smokers, x="age", y="charges", hue="bmi_label", kind="scatter", facet_kws=dict(legend_out=False))
rel_plot.fig.suptitle('Scatterplot of medical charges vs. age for male overweight smokers', fontsize='x-large')
rel_plot.set_axis_labels("Age", "Medical charges", fontsize='large' )
rel_plot._legend.set_title("BMI class",)
plt.show()

#We create a  baseline DataFrame which we'll use to plot our future polynomail model and regularizations
#We set the range from the 'age' variable and create a 1D "grid" between the youngest age and oldest age and we
#populate the space with data in steps of 0.01 "dots"
#Finally we will populate the baseline grid with polynomial features from the 'age' variable
age_list = [g/100 for g in range(DF_male_obese_smokers['age'].min()*100, DF_male_obese_smokers['age'].max()*100)]
DF_plot_1 = pd.DataFrame(columns=['age'])
DF_plot_1['age'] = age_list
for h in range (2, 8):
    Xi = DF_plot_1['age']**h
    Xi_name = "age_degree_" + str(h)
    DF_plot_1.insert(loc = h-1, column=Xi_name, value =Xi)
print(DF_plot_1.head(10))

#We create polynomial regression features. Once again note that this is a less than beautiful way of creating
#Polynomial terms, nut very pedagogic, visual and straightforward
DF_regr = DF_male_obese_smokers[['age', 'charges']].copy(deep=True)
for i in range (2, 8):
    Xi = DF_regr['age']**i
    Xi_name = "age_degree_" + str(i)
    DF_regr.insert(loc = i, column=Xi_name, value =Xi)
print(DF_regr.head(10))   

#We'll use the backwards elimantion selection algorithm from the previous vid to create our predictive polynomail regression model
vars = set(DF_regr.columns)
vars.remove('charges')
Inter_cept = True
pval_threshold = 0.05
highest_pval = 0.10
while (len(vars)>0) and (highest_pval>pval_threshold):
    if Inter_cept==True:
        regr_model = "{} ~ {}".format('charges', ' + '.join(vars))
    else:
        regr_model = "{} ~ {}".format('charges', ' + '.join(vars))
        regr_model = regr_model + "-1"
    print(regr_model)
    testmodel = smf.ols(regr_model, DF_regr).fit()
    pval_testmodel = testmodel.pvalues
    print(pval_testmodel)
    highest_pval = max(pval_testmodel)
    print(highest_pval)
    
    if highest_pval > pval_threshold:
        if pval_testmodel.idxmax()=="Intercept":
            Inter_cept=False
        else:
            vars.remove(pval_testmodel.idxmax())
print(vars)
if Inter_cept==True:
    regr_model = "{} ~ {}".format('charges', ' + '.join(vars))
else:
    regr_model = "{} ~ {}".format('charges', ' + '.join(vars))
    regr_model = regr_model + "-1"
model = smf.ols(regr_model, DF_regr).fit()
print(model.summary())

#We predict data for our model and baseline grid and use the predictions to make a scatter plot for the data and the model
DF_plot_1['Y_pred']=model.predict(DF_plot_1)
fig, ax = plt.subplots()
sns.scatterplot(x=DF_regr.age, y=DF_regr.charges, color='black', ax=ax)
sns.lineplot(x=DF_plot_1.age, y=DF_plot_1.Y_pred, color='red', ax=ax)
plt.title('Data vs. Model predictions', fontsize=20)
plt.xlabel('age', fontsize=18)
plt.ylabel('charges', fontsize=18)
plt.show()

print(DF_regr.head())

#To use Ridge and LAsso regression regularization we need to use sklearn package, And sklearn is not yet easy to use with Pandas
#We must convert DataFrame values to Numpy arrays to continue. From the print(DataFrame.head()) we may learn the
#positions of the variables and a good exercise could be to automate this process
X = DF_regr.iloc[:,[2, 4, 5, 6, 7]]
Y = DF_regr.iloc[:, 1].values.reshape(-1, 1)
X_plot_2 = DF_plot_1.iloc[:,[1, 3, 4, 5, 6]]

#Create 2 Ridge regressions and 2 Lasso regression models
Ridge_regr1 = Ridge(alpha = 0.0000025, max_iter= 100000)
Ridge_regr2 = Ridge(alpha = 200, max_iter=100000)
Lasso_regr1 = Lasso(alpha = 1000, max_iter=100000)
Lasso_regr2 = Lasso(alpha = 0.000250, max_iter= 100000)

#We fit the models to our data.
Ridge_regr1.fit(X, Y)
Ridge_regr2.fit(X, Y)
Lasso_regr1.fit(X, Y)
Lasso_regr2.fit(X, Y)

#We use the models to predict output to display and compare the regularization against our data and unregularized model.
DF_plot_1['Ridge_1_alpha_0_0000025'] = Ridge_regr1.predict(X_plot_2)
DF_plot_1['Ridge_1_alpha_200'] = Ridge_regr2.predict(X_plot_2)
DF_plot_1['Lasso_1_alpha_1000'] = Lasso_regr1.predict(X_plot_2)
DF_plot_1['Lasso_1_alpha_0_000250'] = Lasso_regr2.predict(X_plot_2)

#We graph the models output to compare regularizations with the model and data.
fig, ax = plt.subplots()
sns.scatterplot(x=DF_regr.age, y=DF_regr.charges, color='black', ax=ax)
sns.lineplot(x=DF_plot_1.age, y=DF_plot_1.Y_pred, color='red', ax=ax)
sns.lineplot(x=DF_plot_1.age, y=DF_plot_1.Ridge_1_alpha_0_0000025, color='blue', ax=ax)
sns.lineplot(x=DF_plot_1.age, y=DF_plot_1.Ridge_1_alpha_200, color='green', ax=ax)
sns.lineplot(x=DF_plot_1.age, y=DF_plot_1.Lasso_1_alpha_0_000250, color='yellow', ax=ax)
sns.lineplot(x=DF_plot_1.age, y=DF_plot_1.Lasso_1_alpha_1000, color='orange', ax=ax)
plt.title('Data vs. Model predictions', fontsize=20)
plt.xlabel('age', fontsize=18)
plt.ylabel('charges', fontsize=18)
plt.show()



