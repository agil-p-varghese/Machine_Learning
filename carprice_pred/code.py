import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


df=pd.read_csv('car_data.csv')
#print(df.info())

y=df['price']
X = df.drop(['price', 'CarName'], axis=1)
categorical_features=['fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem']
numerical_features=['car_ID','symboling','wheelbase','carlength','carwidth','carheight','curbweight','enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg']
preprocessor=ColumnTransformer(transformers=[
    ('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),categorical_features),
    ('num',StandardScaler(),numerical_features)
])
pipeline=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',DecisionTreeRegressor(random_state=2))
])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
pipeline.fit(X_train,y_train)
y_pred=pipeline.predict(X_test)
mae=mean_absolute_error(y_pred,y_test)
print("mae ::",mae)
