#mutual information calculation for breast cancer dataset using sklearn
#import required libraries
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()

# Features and target
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate Mutual Information
mi_scores = mutual_info_classif(X_scaled, y, random_state=42)

# Convert to DataFrame for better visualization
mi_df = pd.DataFrame({
    'Feature': data.feature_names,
    'Mutual_Information': mi_scores
})

# Sort features by MI score (descending)
mi_df = mi_df.sort_values(by='Mutual_Information', ascending=False)

# Display Results
print("\nMutual Information Scores (Breast Cancer Dataset):\n")
print(mi_df.to_string(index=False))
