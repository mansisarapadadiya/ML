

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load CSV file
data = pd.read_csv("colors.csv", header=None)

# Convert each row into transaction list
transactions = []
for i in range(len(data)):
    transactions.append([str(item) for item in data.iloc[i].dropna()])

# One-hot encoding
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

df = pd.DataFrame(te_array, columns=te.columns_)

print("Transaction Table:")
print(df)

# Apply Apriori
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Generate Association Rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence']])
