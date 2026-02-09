from flask import Flask, render_template, request, jsonify
from scipy.optimize import differential_evolution
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("models/Tuned_CatBoost.joblib")
    model2=joblib.load("models/Tuned_CatBoostRegressor.joblib")
    print("Model loaded.")
except:
    print("Model loading failed.")
    model = None

# -------- MODEL FEATURE ORDER --------
FEATURE_ORDER = [
    "Binder", "WBRatio", "FA_ratio", "GGBS_ratio",
    "Sand_ratio", "Agg_Binder", "Paste_volume", "age"
]

# ---------------- COST DATA ----------------
COST = {
    'Cement': 6.0,
    'GGBS': 3.6,
    'FlyAsh': 2.0,
    'Water': 0.0,
    'CoarseAggregate': 1.05,
    'Sand': 0.9,
    'Admixture': 45.0  # High cost causes the freezing issue
}

# ---------------- CO2 DATA ----------------
CO2 = {
    'Cement': 1.008,
    'GGBS': 0.064,
    'FlyAsh': 0.026,
    'Water': 0.0003,
    'CoarseAggregate': 0.014,
    'Sand': 0.006,
    'Admixture': 0.72
}

# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template('index.html')


# -------- FEATURE ENGINEERING FUNCTION --------
def build_features(cement, ggbs, flyash, water, coarse, sand, admix, age):
    binder = cement + ggbs + flyash
    if binder == 0:
        return None

    wbr = water / binder
    fa_ratio = flyash / binder
    ggbs_ratio = ggbs / binder
    sand_ratio = sand / (sand + coarse)
    agg_binder = (sand + coarse) / binder
    paste_vol = binder + water + admix 

    row = pd.DataFrame([[binder, wbr, fa_ratio, ggbs_ratio,
                         sand_ratio, agg_binder, paste_vol, age]],
                       columns=FEATURE_ORDER)
    return row


# ---------------- OPTIMIZER ----------------
@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        target_strength = float(request.form['target_strength'])
        age = int(request.form['target_days'])
        has_fa = 'has_fa' in request.form
        has_ggbs = 'has_ggbs' in request.form

        # Bounds (Lower, Upper)
        bounds = [
            (320, 480),   # Cement
            (50, 360),    # GGBS
            (60, 320),    # FlyAsh
            (165, 185),   # Water
            (1100, 1600), # Coarse
            (600, 1000),  # Sand 
            (2.0, 7.0)    # Admix   
        ]

        if not has_ggbs:
            bounds[1] = (0, 0)
        if not has_fa:
            bounds[2] = (0, 0)

        # -------- OBJECTIVE FUNCTION --------
        def objective(x):
            cement, ggbs, flyash, water, coarse, sand, admix = x
            binder = cement + ggbs + flyash
            
            # 1. Hard Constraints
            if binder < 300 or binder > 600:
                return 1e6
            
            # 2. Build Features
            features = build_features(cement, ggbs, flyash, water, coarse, sand, admix, age)
            if features is None:
                return 1e6

            pred_strength = model.predict(features)[0] if model else 0
            
            # Calculate W/B Ratio for logic
            wbr = water / binder

            # 3. Calculate Cost & CO2
            cost = (
                cement*COST['Cement'] + ggbs*COST['GGBS'] + flyash*COST['FlyAsh'] +
                water*COST['Water'] + coarse*COST['CoarseAggregate'] +
                sand*COST['Sand'] + admix*COST['Admixture']
            )

            co2 = (
                cement*CO2['Cement'] + ggbs*CO2['GGBS'] + flyash*CO2['FlyAsh'] +
                water*CO2['Water'] + coarse*CO2['CoarseAggregate'] +
                sand*CO2['Sand'] + admix*CO2['Admixture']
            )

            # 4. PENALTIES

            # A. Strength Penalty
            error = (pred_strength - target_strength) ** 2
            strength_penalty = 0
            if pred_strength < target_strength:
                strength_penalty = 10000 + error * 300 

            # B. Density Constraint (~2400 kg/m3)
            total_mass = cement + ggbs + flyash + water + coarse + sand + admix
            density_penalty = abs(total_mass - 2400) * 80 

            # C. DYNAMIC ADMIXTURE PENALTY (The Fix)
            # ---------------------------------------------------------
            # Rule: Admixture must be at least 0.8% of Binder weight.
            # If W/B is low (<0.38), we need even more (1.1%) for workability.
            
            required_dosage_pct = 0.8 # Default 0.8%
            if wbr < 0.38:
                required_dosage_pct = 1.1 # Increase requirement for high strength mixes

            min_admix_limit = binder * (required_dosage_pct / 100)
            
            admix_penalty = 0
            if admix < min_admix_limit:
                # Strong penalty to force admixture up
                diff = min_admix_limit - admix
                admix_penalty = diff * 5000 
            # ---------------------------------------------------------

            # Minimize Total Score
            return cost + (2.2 * co2) + strength_penalty + density_penalty + admix_penalty

        # -------- RUN OPTIMIZER --------
        result = differential_evolution(
            objective,
            bounds,
            strategy='best1bin',
            maxiter=100,
            popsize=15,
            tol=0.01,
            seed=42
        )

        cement, ggbs, flyash, water, coarse, sand, admix = result.x

        final_features = build_features(cement, ggbs, flyash, water, coarse, sand, admix, age)
        final_strength = model.predict(final_features)[0] if model else 0

        pure_cost = (
            cement*COST['Cement'] + ggbs*COST['GGBS'] + flyash*COST['FlyAsh'] +
            water*COST['Water'] + coarse*COST['CoarseAggregate'] +
            sand*COST['Sand'] + admix*COST['Admixture']
        )

        pure_co2 = (
            cement*CO2['Cement'] + ggbs*CO2['GGBS'] + flyash*CO2['FlyAsh'] +
            water*CO2['Water'] + coarse*CO2['CoarseAggregate'] +
            sand*CO2['Sand'] + admix*CO2['Admixture']
        )
        wbr=water/(cement+ggbs+flyash)
        feat = np.array([[cement, ggbs, flyash, water, coarse, sand, admix, wbr, age]])
        target_strength2 = model2.predict(feat)[0] if model2 else 0
        return jsonify({
            "Cement": round(cement, 1),
            "GGBS": round(ggbs, 1),
            "FlyAsh": round(flyash, 1),
            "Water": round(water, 1),
            "CoarseAggregate": round(coarse, 1),
            "Sand": round(sand, 1),
            "Admix": round(admix, 2), # Admix will now vary based on Binder amount
            "cost": round(pure_cost, 1),
            "co2": round(pure_co2, 1),
            "pred_strength": round(target_strength2, 2)
        })

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)