#import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

#load the auto-mpg dataset
auto_mpg=pd.read_csv("auto-mpg.csv")
auto1=auto_mpg.drop('car name',axis=1)
print(auto1.isnull().sum())

#replace ? with null values
auto1=auto1.replace('?',np.nan)
print('\n print how many null values are there in each column')
print(auto1.isnull().sum())

#convert object datatype to float
auto1['horsepower']=auto1['horsepower'].astype(float)
print('\n datatypes after conversion of horsepower to float')
print(auto1['horsepower'].describe())

print('\n print mean of each column')
print(auto1.mean())

print('\n after fill null value with mean of the column')
auto2=auto1.fillna(auto1.mean())

x=auto2.drop(['mpg'],axis=1)
y=auto2['mpg']

#split the data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#initialize the random forest regressor
rf_regressor=RandomForestRegressor(n_estimators=50,random_state=42)

#train the random forest regressor
rf_regressor.fit(x_train,y_train)

#predict the target variable for test set
y_pred=rf_regressor.predict(x_test)

#calculate the mean squared error(mse) and r^2 score
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mean squqred error:",mse)
print("R^2 score:",r2)
