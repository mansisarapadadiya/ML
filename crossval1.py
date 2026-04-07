from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load data
iris = load_iris()

# Create pipeline (scaling + logistic regression)
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# 3-fold CV
scores = cross_val_score(model, iris.data, iris.target, cv=3)
print("three cross-validation scores: {}".format(scores))
print("average cross-validation score: {:.2f}".format(scores.mean()))

# 5-fold CV
scores = cross_val_score(model, iris.data, iris.target, cv=5)
print("five cross-validation scores: {}".format(scores))
print("average cross-validation score: {:.2f}".format(scores.mean()))
