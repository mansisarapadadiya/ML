#pca on auto-mpg dataset

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#load dataset
df=pd.read_csv("auto-mpg.csv")

#data preprocessing
#replce '?' with NaN
df.replace('?',np.nan,inplace=True)

#convert horsepower to numeric
df['horsepower']=pd.to_numeric(df['horsepower'])

#drop rows with missing values
df.dropna(inplace=True)

#drop non-numeric column(car name)
if 'car name' in df.columns:
    df.drop(columns=['car name'],inplace=True)

#feature selection
#separate features and target (mpg)
x=df.drop(columns=['mpg'])
y=df['mpg']

#convert categorical column 'origin' using one-hot encoding
x=pd.get_dummies(x,columns=['origin'],drop_first=True)

#feature scaling
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

#apply pca
#reduce to 3 principal components
pca=PCA(n_components=3)
x_pca=pca.fit_transform(x_scaled)

#results
print("Original Shape:",x_scaled.shape)
print("Redused Shape:",x_pca.shape)
