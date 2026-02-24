import pandas as pd
from sklearn import datasets
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

#1.load the dataset
iris=datasets.load_iris()
x=iris['data']
y=iris['target']
features=iris['feature_names']

#2.calculate mutual information (mi) scores
#mutual_info_classif estimates MI for discrete target variables
mi_scores=mutual_info_classif(x,y,random_state=42)
mi_df=pd.Series(mi_scores,index=features)

#3.print and visualize the scores
print("mutual information scores:")
print(mi_df.sort_values(ascending=False))

#visualize the feature importance
mi_df.sort_values(ascending=False).plot.bar(figsize=(10,5))
plt.title("feature importance using mutual information")
plt.ylabel("MI score(nats)")
plt.show()



            
