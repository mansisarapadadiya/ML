import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

data=pd.read_csv('pima-indians-diabetes.csv')

#split data into features and target variable
x=data.drop('Outcome',axis=1)
y=data['Outcome']

#split data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#feature scaling
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

#train random forest classifier
rf_classifier=RandomForestClassifier(n_estimators=150,random_state=42)
rf_classifier.fit(x_train_scaled,y_train)

#predictions
y_pred=rf_classifier.predict(x_test_scaled)

#evaluate the model
print("accuracy:",accuracy_score(y_test,y_pred))
print("clasification report:")
print(classification_report(y_test,y_pred))
