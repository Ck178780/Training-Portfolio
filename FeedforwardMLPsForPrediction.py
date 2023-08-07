import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

DataF = pd.read_csv('insurance_data.csv', index_col=0)
print(DataF.head(5))

#plot the dataframe
rel_plot = sns.relplot(data=DataF, x="age", y="charges", kind="scatter", facet_kws=dict(legend_out=False))
rel_plot.fig.suptitle('Scatterplot of medical charges vs. age', fontsize='x-large')
rel_plot.set_axis_labels("Age" , "Medical charges", fontsize='large')
plt.show()

#Classify the BMI values into a select number of medical classes
bmi_labels = ['Healthy weight', 'Overweight', 'Obesity']
cut_bins = [0, 24.900, 29.900, 150.000]
DataF['bmi_label'] = pd.cut(DataF['bmi'], bins=cut_bins, labels=bmi_labels)

#plot age vs. charges data and show the impact of the artificial bmi_label
rel_plot = sns.relplot(data=DataF, x="age", y="charges", hue="bmi_label", kind="scatter", facet_kws=dict(legend_out=False))
rel_plot.fig.suptitle('Scatterplot of medical charges vs. age', fontsize='x-large')
rel_plot.set_axis_labels("Age", "Medical charges", fontsize='large')
rel_plot._legend.set_title("BMI Class",)
plt.show()

#Plot the dataset to ceck for the influence of the smoker feature variable
rel_plot = sns.relplot(data=DataF, x="age", y="charges", hue='smoker', kind="scatter", facet_kws=dict(legend_out=False))
rel_plot.fig.suptitle('Scatterplot of medical charges vs. age for smokers', fontsize='x-large')
rel_plot.set_axis_labels("Age", "Medical charges", fontsize='large')
plt.show()

sc1=StandardScaler()
sc2=StandardScaler()
sc3=StandardScaler()
DataF["charges_scaled"] = sc1.fit_transform(DataF[["charges"]])
DataF["age_scaled"] = sc2.fit_transform(DataF[["age"]])
DataF["bmi_scaled"] = sc3.fit_transform(DataF[["bmi"]])
print(DataF.head())

#Split the DataFrame into 6 parts and make Deepcopies to determine that are separate from the original dataframe.
DF_obe_smo = DataF.loc[((DataF['bmi_label'] == 'Obesity') & (DataF['smoker'] == 'yes'))].copy(deep=True)
DF_owe_smo = DataF.loc[((DataF['bmi_label'] == 'Overweight') & (DataF['smoker'] == 'yes'))].copy(deep=True)
DF_hew_smo = DataF.loc[((DataF['bmi_label'] == 'Healthy weight') & (DataF['smoker'] == 'yes'))].copy(deep=True)
DF_obe_nsm = DataF.loc[((DataF['bmi_label'] == 'Obesity') & (DataF['smoker'] == 'no'))].copy(deep=True)
DF_owe_nsm = DataF.loc[((DataF['bmi_label'] == 'Overweight') & (DataF['smoker'] == 'no'))].copy(deep=True)
DF_hew_nsm = DataF.loc[((DataF['bmi_label'] == 'Healthy weight') & (DataF['smoker'] == 'no'))].copy(deep=True)

#Feature variables are added to a feature dataframe (X1-X6) and a Traget dataframe (Y1-Y6).
X1 = DF_obe_smo[['age_scaled', 'bmi_scaled']]
X2 = DF_owe_smo[['age_scaled', 'bmi_scaled']]
X3 = DF_hew_smo[['age_scaled', 'bmi_scaled']]
X4 = DF_obe_nsm[['age_scaled', 'bmi_scaled']]
X5 = DF_owe_nsm[['age_scaled', 'bmi_scaled']]
X6 = DF_hew_nsm[['age_scaled', 'bmi_scaled']]
Y1 = DF_obe_smo['charges_scaled']
Y2 = DF_owe_smo['charges_scaled']
Y3 = DF_hew_smo['charges_scaled']
Y4 = DF_obe_nsm['charges_scaled']
Y5 = DF_owe_nsm['charges_scaled']
Y6 = DF_hew_nsm['charges_scaled']

#train 1 MLP model for each subset of the data, and predict values
MLP1 = MLPRegressor(random_state=11, max_iter=10000, solver='sgd', hidden_layer_sizes=(5, 5), activation='tanh')
MLP2 = MLPRegressor(random_state=11, max_iter=10000, solver='sgd', hidden_layer_sizes=(6, 6), activation='tanh')
MLP3 = MLPRegressor(random_state=11, max_iter=10000, solver='sgd', hidden_layer_sizes=(6, 6), activation='tanh')
MLP4 = MLPRegressor(random_state=11, max_iter=10000, solver='sgd', hidden_layer_sizes=(6, 6), activation='tanh')
MLP5 = MLPRegressor(random_state=11, max_iter=10000, solver='sgd', hidden_layer_sizes=(6, 6), activation='tanh')
MLP6 = MLPRegressor(random_state=11, max_iter=10000, solver='sgd', hidden_layer_sizes=(6, 6), activation='tanh')
MLP1.fit(X1, Y1)
MLP2.fit(X2, Y2)
MLP3.fit(X3, Y3)
MLP4.fit(X4, Y4)
MLP5.fit(X5, Y5)
MLP6.fit(X6, Y6)
DF_obe_smo['Y_pred'] = MLP1.predict(X1)
DF_owe_smo['Y_pred'] = MLP2.predict(X2)
DF_hew_smo['Y_pred'] = MLP3.predict(X3)
DF_obe_nsm['Y_pred'] = MLP4.predict(X4)
DF_owe_nsm['Y_pred'] = MLP5.predict(X5)
DF_hew_nsm['Y_pred'] = MLP6.predict(X6)

fig, ax =plt.subplots()
sns.scatterplot(x=DF_obe_smo.age_scaled, y=DF_obe_smo.charges_scaled, color='black', ax=ax)
sns.lineplot(x=DF_obe_smo.age_scaled, y=DF_obe_smo.Y_pred, color='black', ax=ax)
sns.scatterplot(x=DF_owe_smo.age_scaled, y=DF_owe_smo.charges_scaled, color='red', ax=ax)
sns.lineplot(x=DF_owe_smo.age_scaled, y=DF_owe_smo.Y_pred, color='red', ax=ax)
sns.scatterplot(x=DF_hew_smo.age_scaled, y=DF_hew_smo.charges_scaled, color='blue', ax=ax)
sns.lineplot(x=DF_hew_smo.age_scaled, y=DF_hew_smo.Y_pred, color='blue', ax=ax)
sns.scatterplot(x=DF_obe_nsm.age_scaled, y=DF_obe_nsm.charges_scaled, color='sienna', ax=ax)
sns.lineplot(x=DF_obe_nsm.age_scaled, y=DF_obe_nsm.Y_pred, color='sienna', ax=ax)
sns.scatterplot(x=DF_owe_nsm.age_scaled, y=DF_owe_nsm.charges_scaled, color='yellow', ax=ax)
sns.lineplot(x=DF_owe_nsm.age_scaled, y=DF_owe_nsm.Y_pred, color='yellow', ax=ax)
sns.scatterplot(x=DF_hew_nsm.age_scaled, y=DF_hew_nsm.charges_scaled, color='lightgreen', ax=ax)
sns.lineplot(x=DF_hew_nsm.age_scaled, y=DF_hew_nsm.Y_pred, color='lightgreen', ax=ax)

plt.title('Data vs. MLP Model predictions', fontsize=20)
plt.xlabel('age_scaled', fontsize=16)
plt.ylabel('charges_scaled', fontsize=16)
plt.show()

#One way to show the predictive advantages of this way handling the MLPs sensitivity to data, is to reassemble the dataframe
#for a residual analysis
Dflist = [DF_obe_smo, DF_owe_smo, DF_hew_smo, DF_obe_nsm, DF_owe_nsm, DF_hew_nsm]
DataF2 = pd.concat(Dflist)

#Restore the scale for the Y_pred predicted variable using the inverse standardscaler. Create residuals
DataF2["Y_pred_unscaled"] = sc1.inverse_transform(DataF2[["Y_pred"]])
DataF2['Residual'] = DataF2['charges'] - DataF2['Y_pred_unscaled']

#Residual plots
fig, ((hist1, res_Y, res_bmi),(res_Y3, sres_X4, sre_X5)) = plt.subplots(2, 3)
DataF2.plot(y='Residual', kind='hist', bins=25, ax=hist1)
DataF2.plot('Y_pred_unscaled', 'Residual', kind='scatter', ax=res_Y)
res_Y.axhline(y=0.0, c='red', linestyle='dashed')
DataF2.plot('bmi', 'Residual', kind='scatter', ax=res_bmi)
res_bmi.axhline(y=0.0, c='red', linestyle='dashed')
DataF2.plot('age', 'Residual', kind='scatter', ax=res_Y3)
res_Y3.axhline(y=0.0, c='red', linestyle='dashed')
DataF2.plot('charges', 'Y_pred_unscaled', kind='scatter', ax=sres_X4)
#sres_X4.axhline(y=0.0, c='red', linestyle='dashed')
DataF2.plot('charges', 'Residual', kind='scatter', ax=sre_X5)
sre_X5.axhline(y=0.0, c='red', linestyle='dashed')
fig.tight_layout()
plt.show()

