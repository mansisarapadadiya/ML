#import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

#load titanic dataset
df=pd.read_csv("titanic.csv")

#dropping irrelevant columns
df=df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

#handling missing values
imputer=SimpleImputer(strategy='most_frequent')
df['Age']=imputer.fit_transform(df[['Age']])
df['Embarked'] = imputer.fit_transform(df[['Embarked']]).ravel()

#encoding categorical variables
label_encoder=LabelEncoder()
df['Sex']=label_encoder.fit_transform(df['Sex'])
df['Embarked']=label_encoder.fit_transform(df['Embarked'])

#splitting data into features and target
x=df.drop('Survived',axis=1)
y=df['Survived']

#spliting data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#standardizing features
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#initializing logistic regression model
log_reg_model = LogisticRegression()

#training the model
log_reg_model.fit(x_train,y_train)

#making predictions
y_pred=log_reg_model.predict(x_test)

#evaluating the model
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

print("\nClassification Report:")
print(classification_report(y_test,y_pred))
