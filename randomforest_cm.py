import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

data=pd.read_csv('pima-indians-diabetes.csv')

#split data into features and target variable
x=data.drop('Outcome',axis=1)
y=data['Outcome']

#train-test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#feature scalling
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

#train random forest classifier
rf_classifier=RandomForestClassifier(n_estimators=150,random_state=42)
rf_classifier.fit(x_train_scaled,y_train)

#predictions
y_pred=rf_classifier.predict(x_test_scaled)

#confusion matrix
cm=confusion_matrix(y_test,y_pred)
print("confusion matrix:",cm)

#evaluate the model
print("accurasy:",accuracy_score(y_test,y_pred))

