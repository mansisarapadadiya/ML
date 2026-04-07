# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# load dataset
df = pd.read_csv('spambase.csv')

# clean column names
df.columns = df.columns.str.strip()

# features and target
X = df.drop('spam', axis=1)
y = df['spam']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scale data (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Models


# Naive Bayes (Gaussian for numeric data)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

# Decision Tree - ID3
id3_model = DecisionTreeClassifier(criterion='entropy')
id3_model.fit(X_train, y_train)
id3_pred = id3_model.predict(X_test)

# Decision Tree - CART
cart_model = DecisionTreeClassifier(criterion='gini')
cart_model.fit(X_train, y_train)
cart_pred = cart_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)


# Evaluation

def evaluate(name, y_test, y_pred):
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

evaluate("Naive Bayes", y_test, nb_pred)
evaluate("Decision Tree (ID3)", y_test, id3_pred)
evaluate("Decision Tree (CART)", y_test, cart_pred)
evaluate("Random Forest", y_test, rf_pred)
evaluate("SVM", y_test, svm_pred)
