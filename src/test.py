import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("data/features.csv")

X = df[['Cement', 'GGBS', 'FlyAsh', 'Water', 'CoarseAggregate', 'Sand', 'Admixture', 'WBRatio', 'age']]
y = df['Strength']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models={
    "AdaBoost":"models\AdaBoost_pipeline.joblib",
    "Random Forest":"models\RandomForest_pipeline.joblib",
    "Cat_Boost":"models\CatBoost_pipeline.joblib",
    "XGBoost":"models\XGBoost_pipeline.joblib"
}

table=PrettyTable()
table.field_names=["Model","MAE","RMSE","R2"]
res={}

for name,path in models.items():
    model=joblib.load(path)
    y_pred=model.predict(X_test)
    mae=mean_absolute_error(y_test,y_pred)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    r2=r2_score(y_test,y_pred)
    res[name]=mae
    table.add_row([name,round(mae,3),round(rmse,3),round(r2,3)])

print("Model Evaluation Results:")
print(table)

best_model=min(res,key=res.get)
print(f"Best model based on MAE:{best_model}")

tuned_model=joblib.load("models/Tuned_XGBoost.joblib")
y_pred_tuned=tuned_model.predict(X_test)
r2_score1=r2_score(y_test,y_pred_tuned)
print(f"Tuned XGBoost R2 Score: {r2_score1:.4f}")

tuned_model=joblib.load("models/Tuned_CatBoost.joblib")
y_pred_tuned=tuned_model.predict(X_test)
r2_score1=r2_score(y_test,y_pred_tuned)
print(f"Tuned CatBoost R2 Score: {r2_score1:.4f}")