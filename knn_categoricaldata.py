# Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset from CSV
df = pd.read_csv("knn_dataset.csv")

# Display first few rows (optional)
print(df.head())

# Convert categorical columns to numeric using Label Encoding
le = LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

# Separate features and target
X = df.drop("Job_offered", axis=1)
y = df["Job_offered"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Train model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("KNN Accuracy:", accuracy)
