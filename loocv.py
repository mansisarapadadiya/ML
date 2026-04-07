from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load dataset
iris = load_iris()

print("iris labels:")
print(iris.target)

# Create LOOCV object
loo = LeaveOneOut()

# Create pipeline (scaling + logistic regression)
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# Perform LOOCV
scores = cross_val_score(model, iris.data, iris.target, cv=loo)

print("number of cv iterations:", len(scores))
print("mean accuracy: {:.2f}".format(scores.mean()))
