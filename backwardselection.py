import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#load and clean dataset

df=pd.read_csv("auto-mpg.csv")

df.replace('?',np.nan,inplace=True)

df['horsepower']=pd.to_numeric(df['horsepower'])
df=df.dropna()

if'car name' in df.columns:
    df=df.drop(columns=['car name'])

#define features and target

x=df.drop(columns=['mpg'])
y=df['mpg']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#backward feature selection

selected_features=list(x.columns)
model=LinearRegression()
model.fit(x_train[selected_features],y_train)
y_pred=model.predict(x_test[selected_features])
best_score=r2_score(y_test,y_pred)


print("Backward Feature Elimination Process:\n")
print(f"Initial R2 Score (All Features): {best_score:.4f}\n")

while len(selected_features) > 1:
    
    scores = []
    
    for feature in selected_features:
        features_to_test = selected_features.copy()
        features_to_test.remove(feature)
        
        model = LinearRegression()
        model.fit(x_train[features_to_test], y_train)
        
        y_pred = model.predict(x_test[features_to_test])
        score = r2_score(y_test, y_pred)
        
        scores.append((score, feature))
    
    # Find removal that gives highest R2
    scores.sort(reverse=True)
    current_best_score, worst_feature = scores[0]
    
    if current_best_score >= best_score:
        best_score = current_best_score
        selected_features.remove(worst_feature)
        
        print(f"Removed: {worst_feature}, R2 Score: {best_score:.4f}")
    else:
        break

print("\nFinal Selected Features:")
print(selected_features)
