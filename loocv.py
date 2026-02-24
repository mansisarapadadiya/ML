from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut

iris = load_iris()
print("iris labels:\n{}".format(iris.target))
logreg = LogisticRegression()

loo = LeaveOneOut()
scores = cross_val_score(logreg,iris.data,iris.target,cv=loo)
print("number of cv iterations:",len(scores))
print("mean accuracy: {:.2f}".format(scores.mean()))
