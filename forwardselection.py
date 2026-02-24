import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df=pd.read_csv("D:/7005/ML/auto-mpg.csv")
df.replace('?',np.nan,inplace=True)

df['horsepower']=pd.to_numeric(df['horsepower'])
df=df.dropna()

if'car name'in df.columns:
    df=df.drop(columns=['car name'])

x=df.drop(columns=['mpg'])
y=df['mpg']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#remaining_features-> features not yet selected
#selected_features-> features already selected
#best_score-> best R score achieved so far
#we initialize best_score=- so that the first feature always improves the score

remaining_features=list(x.columns)
selected_features=[]
best_score=-np.inf

print("forward feature selection process:\n")

#the loop continues untill:
#no features remain,or
#adding a new feature does not improve performance

while remaining_features:
    scores=[]

    #temporarly add one feature at a time to the selected set.
    for feature in remaining_features:
        features_to_test = selected_features+[feature]

        #train regression model using only the selected+candidate feature.
        #this is the wrapper mechanism)(model-based evaluation)

        model=LinearRegression()
        model.fit(x_train[features_to_test],y_train)

        y_pred=model.predict(x_test[features_to_test])
        score=r2_score(y_test,y_pred)

        scores.append((score,feature))

    #sort features by R² score.
    #Select feature giving highest improvement

    scores.sort(reverse=True)
    current_best_score,best_feature=scores[0]

#If adding the feature improves R²:
#Update best_score
#Add feature permanently
#Remove from remaining list
#Else:
#Stop algorithm (no further improvement)

    if current_best_score > best_score:
        best_score=current_best_score
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

        print(f"added: {best_feature},r2 score: {best_score:.4f}")
    else:
        break
print("\nfinal selected features:")
print(selected_features)
        
    
