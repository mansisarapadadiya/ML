#decision tree using id3(entropy,gini)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

data=pd.read_csv('pima-indians-diabetes.csv')

#split data into features and target variable
x=data.drop('Outcome',axis=1)
y=data['Outcome']

#train-test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#feature scalling
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

#create decision tree using entropy(id3)
#model=decisiontreeclassifier(criterion='entropy')

#create decision tree using cart(gini)
model=DecisionTreeClassifier(criterion='gini')

#train model
model.fit(x_train,y_train)

#predictions
y_pred=model.predict(x_test)

#confusion matrix
cm=confusion_matrix(y_test,y_pred)
print("confusion matrix:",cm)

#accuracy
accuracy=accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)
