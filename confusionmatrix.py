import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

data=load_breast_cancer()
x=data.data
y=data.target

x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42,stratify=y)

model=LogisticRegression(max_iter=5000)
model.fit(x_train,y_train)

#predictions and confusion matrix

y_pred=model.predict(x_test)
cm=confusion_matrix(y_test,y_pred)

print("confusion matrix:\n",cm)

tn,fp,fn,tp=cm.ravel()

print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")

#another way to extract tn,fp,fn,tp

TN= cm[0,0]
FP= cm[0,1]
FN= cm[1,0]
TP= cm[1,1]

print("\nExtracted values:")
print("TP:",TP)
print("TN:",TN)
print("FP:",FP)
print("FN:",FN)
