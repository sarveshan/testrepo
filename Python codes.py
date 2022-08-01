# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 22:33:20 2022

@author: User
"""
#IMPORTING PACKAGES & DATASET

#packages
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.metrics import explained_variance_score
from sklearn import metrics
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
import statsmodels.stats.diagnostic as dg
import numpy as np
from sklearn.metrics import mean_squared_error

#dataset
mydata = pd.read_csv("C:\\Users\\User\\Downloads\\Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System (1).csv")
mydata.info()

#DATA PREPROCESSSING

#cleaning data
#removing unwanted variables for this study
df = mydata.drop(['YearEnd','Class','Topic','Data_Value_Unit','Datasource','ClassID',
             'Low_Confidence_Limit','TopicID','DataValueTypeID',
             'Data_Value_Type','Data_Value_Footnote_Symbol','Data_Value_Footnote',
             'StratificationCategoryId1','StratificationID1','QuestionID'],1)

#removing data from all other years except for 2020
df1 = df[(df.YearStart != 2011) & (df.YearStart != 2012) & 
         (df.YearStart != 2013) & (df.YearStart != 2014) & 
         (df.YearStart != 2015) & (df.YearStart != 2016) & 
         (df.YearStart != 2017) & (df.YearStart != 2018) & 
         (df.YearStart != 2019)]

#removing data from states that are not included in the 50 states of USA 
finaldata = df1[(df1.LocationAbbr != 'GU') & (df1.LocationAbbr != 'PR') & 
                (df1.LocationAbbr != 'US') & (df1.LocationAbbr != 'VI')]

#focusing on the obesity question and removing all other questions
finaldata1 = finaldata[(finaldata.Question == 'Percent of adults aged 18 years and older who have obesity')]
                     
#creating new dataset for education
finaldata_edu = finaldata1[finaldata1.StratificationCategory1 == 'Education']
finaldata_edu=finaldata_edu[['YearStart','LocationDesc','Data_Value','Education']]   

#creating a list of 4 educational levels
edulevel = finaldata_edu.Education.unique()

#assinging dummy variables
for i in edulevel:
    finaldata_edu[i]=finaldata['Education'].apply(lambda x: int(x==i))

#dropping the YearStart column since all values are now focused on the year 2020
#dropping the LocationDesc since we are not focusing on the location
finaldata_edu_data_only = finaldata_edu.drop(['YearStart','LocationDesc'],1) 

#renaming the columns to a shorter version
finaldata_edu_data_only.rename({"Data_Value":"Obesity","High school graduate":"HSG","Less than high school":"LHS",
                                "Some college or technical school":"SCT","College graduate":"CG"},axis=1,inplace=True)

#since there are 4 dummy variables,we are only including 3 coefficients, allowing SCT to become the reference group
columns1 = finaldata_edu_data_only.columns[0:5]

#CORRELATION MATRIX

#producing the correlation matrix
corr_matrix = finaldata_edu_data_only[columns1].corr()
print(corr_matrix)

#generating the heatmap for the correlation matrix
sns.heatmap(corr_matrix, annot=True)
plt.show()

#PLOTS

#plotting the total counts of each education level
finaldata_edu_data_only['Education'].value_counts().plot(kind='bar')

#Boxplot
#dropping columns that are not needed for the plot
data_and_edulevel = finaldata_edu_data_only.drop(['HSG','LHS','SCT','CG'],1)
#Replacing long names to shorter names
data_and_edulevel['Education'].replace(to_replace=['High school graduate','Less than high school','Some college or technical school','College graduate'],value=['HSG','LHS','SCT','CG'],inplace=True)
#producing boxplot 
sns.catplot(x="Education",y="Obesity",kind="box",data=data_and_edulevel)

#LINEAR REGRESSION

#Method 1: Using Scikit-learn
#creating regression model
model = LinearRegression()
columns = finaldata_edu.columns[5:8]
X = finaldata_edu[columns]
#preparing variables
X_std = StandardScaler().fit_transform(X)
y = finaldata_edu['Data_Value']
y = y.to_numpy()
#splitting dataset randomly into 82% training data and 18% test data
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.18, 
                                                    random_state=42)
#fitting the traing data into the model 
model.fit(X_train,y_train)    
#print coefficients and intercept
print('Model Coefficients',model.coef_)
print('Model Coefficients',model.intercept_)

#Method 2:Using Stasmodel
#adding a constant so that the regression line fits the intercept,
#minimizing the bias
X_train2 = sm.add_constant(X_train)
#creating a regression model and fit with the existing data
test_model = sm.OLS(y_train,X_train2).fit()
print_model = test_model.summary()
print(print_model)

#PREDICTION

#predicting values using test set
y_pred = model.predict(X_test)

#EVALUATION OF THE MODEL

#Checking for heteroscedasticity and autocorrelation
#checking for heteroscedasticity using Goldfeld Quandt
name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(test_model.resid,
test_model.model.exog)
print('\nGoldfeld Quandt Test: ', lzip(name,test))
#checking for autocorrelation via Breusch-Godfrey Test
print('\nBreusch-Godfrey Test: ',dg.acorr_breusch_godfrey(test_model, nlags=3))

#Evaluation of error within the model
#Mean Absolute Error (MAE)
print('\nMean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
#Mean Squared Error (MSE)
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))
#Root Mean Squared Error of model(RMSE)
print('\nRoot Mean Squared Error (RMSE): ', (np.sqrt(mean_squared_error(y_test, y_pred))))
#RMSE of training and testing data
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
print("Root Mean Squared Error of Training Set: {}".format(rmse_train))
print("Root Mean Squared Error of Testing Set: {}".format(rmse_test))
#Mean Absolute Percentage Error (MAPE)
def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100
actual = y_test
pred = y_pred
print('\nMean Absolute Percentage Error (MAPE):',mape(actual, pred))

#Variance and fitness of the model
#R squared on training data
print('R^2 on training...',model.score(X_train,y_train))
#R squared in testing data
print('R^2 on test...',model.score(X_test,y_test))
#Explained Variance Score
print('Explained Variance Score...',explained_variance_score(y_test,y_pred))




















                                             
