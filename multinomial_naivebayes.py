#import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#load iris dataset
data=load_breast_cancer()
x=data.data
y=data.target

#split the data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#initialize gaussian naive bayes classifier
naive_bayes=MultinomialNB()

#train the classifier
naive_bayes.fit(x_train,y_train)

#predict the labels for test data
y_pred=naive_bayes.predict(x_test)

#calculate accuracy
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
