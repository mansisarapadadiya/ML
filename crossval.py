from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris=load_iris()
logreg = LogisticRegression()
scores = cross_val_score(logreg,iris.data,iris.target,cv=3)
print("three cross-validation scores: {}".format(scores))
print("average cross-validation score: {:.2f}".format(scores.mean()))

scores = cross_val_score(logreg,iris.data,iris.target,cv=5)
print("five cross-validation scores: {}".format(scores))
print("average cross-validation score: {:.2f}".format(scores.mean()))









'''#output: three cross-validation scores: [0.98 0.96 0.98]
average cross-validation score: 0.97

five cross-validation scores: [0.96666667 1.         0.93333333 0.96666667 1.        ]
average cross-validation score: 0.97'''
