import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error

#load dataset
data=pd.read_csv("linearregressiondataset.csv")

#define x and y
x=data[['Population']]
y=data['Profit']

#fit the model
model=LinearRegression()
model.fit(x,y)

#prediction
y_pred=model.predict(x)

#coefficient&intercept
print("Coefficient",model.coef_[0])
print("intercept",model.intercept_)

#calculate MAE,MSE,RMSE
mae=mean_absolute_error(y,y_pred)
mse=mean_squared_error(y,y_pred)
rmse=np.sqrt(mse)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)

#visualization

plt.scatter(x,y,color='blue',label="actual data")
plt.plot(x,y_pred,color='red',label="Regression Line")

plt.xlabel("Population")
plt.ylabel("Profit")
plt.title("simple linear regression")
plt.legend()
plt.show()
