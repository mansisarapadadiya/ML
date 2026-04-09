import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Using an in-built dataset: Grocery transactions (each row represents items purchased together)
dataset = [['milk', 'bread', 'butter'],
           ['bread', 'butter'],
           ['milk', 'bread', 'butter', 'cheese'],
           ['milk', 'bread'],
           ['butter', 'cheese'],
           ['bread', 'butter', 'cheese']]

# Step 1: Convert the dataset into a one-hot encoded DataFrame
te = TransactionEncoder()
te_data = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_data, columns=te.columns_)

# Step 2: Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# Step 3: Generate association rules from the frequent itemsets
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.75)

# Output the results
print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules)
