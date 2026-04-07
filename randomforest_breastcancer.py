#import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

#load the breast cancer dataset
cancer=datasets.load_breast_cancer()
x=cancer.data
y=cancer.target

#split the data into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#initialize the random forest classifier
rf_classifier=RandomForestClassifier(n_estimators=100,random_state=42)

#train the random forest classifier
rf_classifier.fit(x_train,y_train)

#predict the labels for test set
y_pred=rf_classifier.predict(x_test)

#claculate the accuracy
accuracy=accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)

#print classification report
print("\nClassification Report:")
print(classification_report(y_test,y_pred))
                                            
