#importing libraries

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#load the california housing prices datasets
data=fetch_california_housing()
print(data.DESCR)
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target

#split the data into features(x) and target variable(y)
x=df.drop(columns=['target'])
y=df['target']

#split the data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#create and train the linear regression model
model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

#calculate evaluation metrics
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r_squared=r2_score(y_test,y_pred)
rmse=np.sqrt(mse)

#print the evaluation metrics
print("mean absolute error(mae):",mae)
print("mean squared error(mse):",mse)
print("r-squared(r):",r_squared)
print("root mean squared error (rmse):",rmse)
