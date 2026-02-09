import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import joblib
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score


df = pd.read_csv(r"data\features.csv")

X = df[["Binder","WBRatio","FA_ratio","GGBS_ratio","Sand_ratio","Agg_Binder","Paste_volume", 'age']]
y = df['Strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Initial Model Comparison ---
models = {
    "RandomForest": Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    "XGBoost": Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42))
    ]),
    "AdaBoost": Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', AdaBoostRegressor(n_estimators=100))
    ]),
    "CatBoost": Pipeline([ 
        ('scaler', StandardScaler()), 
        ('regressor', CatBoostRegressor(verbose=0, iterations=500)) 
    ])
}

results = []
print("Training Initial Models...")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    plt.figure(figsize=(6,6))

    plt.scatter(y_test, preds, alpha=0.6)

    # 45° perfect prediction line
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

    plt.xlabel("True Strength")
    plt.ylabel("Predicted Strength")
    plt.title(f"{name} Regression Performance")
    plt.grid(True)


    plt.text(min_val, max_val*0.95,
         f"R² = {r2:.3f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}",
         fontsize=11,
         bbox=dict(facecolor='white', alpha=0.6))

    plt.show()

    results.append({"Model": name, "R2": round(r2, 4), "MAE": round(mae, 4), "RMSE": round(rmse, 4)})
    joblib.dump(model, f"models/{name.replace(' ', '_')}_pipeline.joblib")

    print(f" > {name} trained.")


results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
print("\n--- Model Performance Comparison ---")
print(results_df)



# Hyperparameter tuning
# ================= MODEL CONFIGURATION =================

model_configs = {
    "CatBoost": {
        "estimator": CatBoostRegressor(verbose=0, random_state=42),
        "param_grid": {
            'regressor__iterations': [300, 500, 800],
            'regressor__depth': [4, 6, 8],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__l2_leaf_reg': [1, 3, 5, 7]
        },
        "random_iter": 15
    },

    "XGBoost": {
        "estimator": XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        ),
        "param_grid": {
            'regressor__n_estimators': [300, 600],
            'regressor__max_depth': [4, 6],
            'regressor__learning_rate': [0.03, 0.1],
            'regressor__subsample': [0.8, 1.0],
            'regressor__colsample_bytree': [0.8, 1.0]
        },
        "random_iter": 25
    }
}


# ================= PIPELINE CREATOR =================

def create_pipeline(model, model_name):
    if model_name == "CatBoost":
        return Pipeline([
            ('regressor', model)  # No scaling for CatBoost
        ])
    else:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])


# ================= SEARCH FUNCTION =================

def run_search(model_name, config, X_train, y_train):
    results = []
    param_grid = deepcopy(config["param_grid"])   # Prevent mutation
    pipeline = create_pipeline(config["estimator"], model_name)

    # -------- GRID SEARCH --------
    print(f"\n🔍 Grid Search — {model_name}")
    start = time.time()

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring='r2',
        error_score='raise',
        verbose=1
    )
    grid.fit(X_train, y_train)

    results.append({
        "Model": model_name,
        "Method": "Grid Search",
        "Best R2": grid.best_score_,
        "Time (s)": time.time() - start
    })

    # -------- RANDOM SEARCH --------
    print(f"\n🎲 Random Search — {model_name}")
    start = time.time()

    random = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=config["random_iter"],
        cv=3,
        n_jobs=-1,
        scoring='r2',
        random_state=42,
        error_score='raise',
        verbose=1
    )
    random.fit(X_train, y_train)

    results.append({
        "Model": model_name,
        "Method": "Random Search",
        "Best R2": random.best_score_,
        "Time (s)": time.time() - start
    })

    # -------- SAVE BEST MODEL --------
    best_model = grid.best_estimator_
    path = f"models/Tuned_{model_name}.joblib"
    joblib.dump(best_model, path)

    print(f"✅ {model_name} best model saved → {path}")
    print(f"Best Parameters:\n{grid.best_params_}")

    return results


# ================= RUN ALL MODELS =================

all_results = []

for model_name, config in model_configs.items():
    all_results.extend(run_search(model_name, config, X_train, y_train))

comparison_df = pd.DataFrame(all_results)

print("\n📊 --- Overall Hyperparameter Tuning Comparison ---")
print(comparison_df)
