from flask import Flask, render_template, request, jsonify
from scipy.optimize import differential_evolution
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load(r"models\Tuned_CatBoost.joblib")
    print("✅ Model loaded.")
except:
    print("❌ Model loading failed.")
    model = None

# Feature order must EXACTLY match training
FEATURE_ORDER = [
    'Cement', 'GGBS', 'FlyAsh', 'Water',
    'CoarseAggregate', 'Sand', 'Admixture',
    'WBRatio', 'age'
]

# ---------------- COST DATA ----------------
COST = {
    'Cement': 6.0,
    'GGBS': 3.6,
    'FlyAsh': 2.0,
    'Water': 0.0,
    'CoarseAggregate': 1.05,
    'Sand': 0.9,
    'Admixture': 45.0
}

# ---------------- CO2 DATA (kg CO₂ per kg) ----------------
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

# ---------------- OPTIMIZER ----------------
@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        target_strength = float(request.form['target_strength'])
        age = int(request.form['target_days'])
        has_fa = 'has_fa' in request.form
        has_ggbs = 'has_ggbs' in request.form

        # Bounds: Cement, GGBS, FlyAsh, Water, Coarse, Sand, Admix
        bounds = [
            (310, 450),    # Cement
            (50, 360),      # GGBS
            (60, 320),      # FlyAsh
            (160, 180),    # Water
            (1100, 1400),  # CoarseAggregate
            (700, 1000),    # Sand
            (2.0, 4.5)     # Admixture
        ]

        if not has_ggbs:
            bounds[1] = (0, 0)
        if not has_fa:
            bounds[2] = (0, 0)

        # -------- OBJECTIVE FUNCTION --------
        def objective(x):
            cement, ggbs, flyash, water, coarse, sand, admix = x
            binder = cement + ggbs + flyash

            # ---- REALISTIC CONCRETE CONSTRAINTS ----
            if binder < 300 or binder > 500:
                return 1e6
            if water / binder < 0.30 or water / binder > 0.60:
                return 1e6

            wbr = water / binder

            row = pd.DataFrame([[cement, ggbs, flyash, water, coarse, sand, admix, wbr, age]],
                               columns=FEATURE_ORDER)

            pred_strength = model.predict(row)[0] if model else 0

            # --- COST ---
            cost = (
                cement*COST['Cement'] +
                ggbs*COST['GGBS'] +
                flyash*COST['FlyAsh'] +
                water*COST['Water'] +
                coarse*COST['CoarseAggregate'] +
                sand*COST['Sand'] +
                admix*COST['Admixture']
            )

            # --- CO2 ---
            co2 = (
                cement*CO2['Cement'] +
                ggbs*CO2['GGBS'] +
                flyash*CO2['FlyAsh'] +
                water*CO2['Water'] +
                coarse*CO2['CoarseAggregate'] +
                sand*CO2['Sand'] +
                admix*CO2['Admixture']
            )

            # ---- Strength penalty ----
            if pred_strength < target_strength:
                penalty = 8000 + (target_strength - pred_strength)**2 * 120
                return cost + penalty

            # ---- Multi-objective (Cost + Carbon) ----
            score = cost + 2.2 * co2
            return score

        # -------- RUN OPTIMIZER --------
        result = differential_evolution(
            objective,
            bounds,
            strategy='best1bin',
            maxiter=60,
            popsize=15,
            tol=0.01,
            seed=42,
            updating='deferred'
        )

        x = result.x
        cement, ggbs, flyash, water, coarse, sand, admix = x
        binder = cement + ggbs + flyash
        wbr = water / binder

        final_row = pd.DataFrame([[cement, ggbs, flyash, water, coarse, sand, admix, wbr, age]],
                                 columns=FEATURE_ORDER)

        final_strength = model.predict(final_row)[0]

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

        return jsonify({
            "Cement": round(cement, -1),
            "GGBS": round(ggbs, -1),
            "FlyAsh": round(flyash, -1),
            "Water": round(water, -1),
            "CoarseAggregate": round(coarse, -1),
            "Sand": round(sand, -1),
            "Admix": round(admix, 2),
            "cost": round(pure_cost, 1),
            "co2": round(pure_co2, 1),
            "pred_strength": round(final_strength, 2)
        })

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
