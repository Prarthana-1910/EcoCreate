from flask import Flask, render_template, request, jsonify
from scipy.optimize import differential_evolution
import pandas as pd
import joblib

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("models/ConcreteAI_XGBoost_Best.joblib")
    print("Model loaded.")
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None

# -------- MODEL FEATURE ORDER --------
FEATURE_ORDER = [
    "Cement_kg_m3", "Fly_Ash_kg_m3", "GGBS_kg_m3", "metakolin_kg_m3",
    "Water_kg_m3", "Sand_kg_m3", "AGE", "admixture", "Coarse aggregate",
    "SCMContent", "Binder", "WBRatio", "AggregateToBinder", "AdmixtureToBinder",
]

# ---------------- COST DATA (per kg) ----------------
COST = {
    "Cement": 6.0,
    "GGBS": 3.6,
    "FlyAsh": 2.0,
    "Metakaolin": 8.0,
    "Water": 0.1,
    "CoarseAggregate": 1.05,
    "Sand": 0.9,
    "Admixture": 45.0,
}

# ---------------- CO2 DATA (kg CO2 per kg material) ----------------
CO2 = {
    "Cement": 1.008,
    "GGBS": 0.064,
    "FlyAsh": 0.026,
    "Metakaolin": 0.33,
    "Water": 0.0003,
    "CoarseAggregate": 0.014,
    "Sand": 0.006,
    "Admixture": 0.72,
}


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


# -------- FEATURE ENGINEERING --------
def build_features(cement, flyash, ggbs, metakaolin, water, coarse, sand, admix, age):
    scm_content = flyash + ggbs + metakaolin
    binder = cement + scm_content

    if binder <= 0:
        return None

    wb_ratio = water / binder
    aggregate_to_binder = (sand + coarse) / binder
    admixture_to_binder = admix / binder

    row = pd.DataFrame([[
        float(cement),
        float(flyash),
        float(ggbs),
        float(metakaolin),
        float(water),
        float(sand),
        float(age),
        float(admix),
        float(coarse),
        float(scm_content),
        float(binder),
        float(wb_ratio),
        float(aggregate_to_binder),
        float(admixture_to_binder)
    ]], columns=FEATURE_ORDER)

    return row[FEATURE_ORDER].astype(float)


# -------- GRADE-SPECIFIC RULES --------
def get_grade_config(target_strength: float):
    if target_strength < 40:
        return {
            "tcm_range": (360, 400),
            "cement_pct_range": (0.50, 0.56),
            "fa_pct_range": (0.18, 0.24),
            "ggbs_pct_range": (0.14, 0.20),
            "mk_pct_range": (0.03, 0.05),
            "water_bounds": (175, 205),
            "admix_bounds": (1.5, 5.0),
            "sand_bounds": (620, 860),
            "coarse_bounds": (1000, 1420),
            "wb_max": 0.58,
            "binder_range": (360, 420),
            "sand_ratio_target": 0.38,
        }

    if 40 <= target_strength < 50:
        return {
            "tcm_range": (400, 450),
            "cement_pct_range": (0.47, 0.50),
            "fa_pct_range": (0.23, 0.27),
            "ggbs_pct_range": (0.20, 0.22),
            "mk_pct_range": (0.05, 0.07),
            "water_bounds": (170, 195),
            "admix_bounds": (2.0, 6.0),
            "sand_bounds": (640, 860),
            "coarse_bounds": (1000, 1420),
            "wb_max": 0.55,
            "binder_range": (400, 460),
            "sand_ratio_target": 0.38,
        }

    if 50 <= target_strength < 60:
        return {
            "tcm_range": (450, 500),
            "cement_pct_range": (0.48, 0.50),
            "fa_pct_range": (0.23, 0.27),
            "ggbs_pct_range": (0.20, 0.22),
            "mk_pct_range": (0.07, 0.09),
            "water_bounds": (160, 185),
            "admix_bounds": (3.0, 8.0),
            "sand_bounds": (620, 840),
            "coarse_bounds": (1010, 1400),
            "wb_max": 0.45,
            "binder_range": (450, 510),
            "sand_ratio_target": 0.37,
        }

    if 60 <= target_strength < 70:
        return {
            "tcm_range": (500, 530),
            "cement_pct_range": (0.46, 0.48),
            "fa_pct_range": (0.18, 0.22),
            "ggbs_pct_range": (0.25, 0.28),
            "mk_pct_range": (0.08, 0.11),
            "water_bounds": (150, 175),
            "admix_bounds": (4.0, 10.0),
            "sand_bounds": (600, 820),
            "coarse_bounds": (1020, 1380),
            "wb_max": 0.40,
            "binder_range": (500, 540),
            "sand_ratio_target": 0.36,
        }

    if 70 <= target_strength < 80:
        return {
            "tcm_range": (530, 560),
            "cement_pct_range": (0.44, 0.47),
            "fa_pct_range": (0.14, 0.19),
            "ggbs_pct_range": (0.26, 0.28),
            "mk_pct_range": (0.09, 0.12),
            "water_bounds": (145, 170),
            "admix_bounds": (5.0, 12.0),
            "sand_bounds": (580, 800),
            "coarse_bounds": (1030, 1360),
            "wb_max": 0.38,
            "binder_range": (530, 570),
            "sand_ratio_target": 0.35,
        }

    if 80 <= target_strength < 90:
        return {
            "tcm_range": (560, 590),
            "cement_pct_range": (0.44, 0.47),
            "fa_pct_range": (0.13, 0.17),
            "ggbs_pct_range": (0.26, 0.28),
            "mk_pct_range": (0.12, 0.15),
            "water_bounds": (140, 165),
            "admix_bounds": (6.0, 14.0),
            "sand_bounds": (560, 780),
            "coarse_bounds": (1040, 1340),
            "wb_max": 0.35,
            "binder_range": (560, 600),
            "sand_ratio_target": 0.34,
        }

    return {
        "tcm_range": (560, 590),
        "cement_pct_range": (0.44, 0.47),
        "fa_pct_range": (0.13, 0.17),
        "ggbs_pct_range": (0.26, 0.28),
        "mk_pct_range": (0.12, 0.15),
        "water_bounds": (138, 160),
        "admix_bounds": (6.5, 15.0),
        "sand_bounds": (540, 760),
        "coarse_bounds": (1040, 1320),
        "wb_max": 0.33,
        "binder_range": (560, 600),
        "sand_ratio_target": 0.33,
    }


def midpoint(rng):
    return (rng[0] + rng[1]) / 2.0


def compute_selected_scm_targets(cfg, has_fa, has_ggbs, has_metakaolin):
    fa_mid = midpoint(cfg["fa_pct_range"])
    ggbs_mid = midpoint(cfg["ggbs_pct_range"])
    mk_mid = midpoint(cfg["mk_pct_range"])

    selected = {}
    if has_fa:
        selected["fa"] = fa_mid
    if has_ggbs:
        selected["ggbs"] = ggbs_mid
    if has_metakaolin:
        selected["mk"] = mk_mid

    total_selected_nominal = sum(selected.values())

    if total_selected_nominal <= 1e-9:
        return {
            "target_total_scm": 0.0,
            "fa_share": 0.0,
            "ggbs_share": 0.0,
            "mk_share": 0.0,
        }

    cement_mid = midpoint(cfg["cement_pct_range"])
    target_total_scm = 1.0 - cement_mid

    return {
        "target_total_scm": target_total_scm,
        "fa_share": selected.get("fa", 0.0) / total_selected_nominal,
        "ggbs_share": selected.get("ggbs", 0.0) / total_selected_nominal,
        "mk_share": selected.get("mk", 0.0) / total_selected_nominal,
    }


# ---------------- OPTIMIZER ----------------
@app.route("/optimize", methods=["POST"])
def optimize():
    try:
        target_strength = float(request.form["target_strength"])
        age = 28

        has_fa = "has_fa" in request.form
        has_ggbs = "has_ggbs" in request.form
        has_metakaolin = "has_metakaolin" in request.form

        if model is None:
            raise Exception("Model not loaded. Check model file path.")

        cfg = get_grade_config(target_strength)
        scm_targets = compute_selected_scm_targets(cfg, has_fa, has_ggbs, has_metakaolin)

        tcm_min, tcm_max = cfg["tcm_range"]
        cement_pct_min, cement_pct_max = cfg["cement_pct_range"]

        cement_bounds = (
            tcm_min * cement_pct_min,
            tcm_max * cement_pct_max
        )

        water_bounds = cfg["water_bounds"]
        admix_bounds = cfg["admix_bounds"]
        sand_bounds = cfg["sand_bounds"]
        coarse_bounds = cfg["coarse_bounds"]

        if has_fa:
            fa_bounds = (
                max(20.0, tcm_min * cfg["fa_pct_range"][0] * 0.85),
                tcm_max * cfg["fa_pct_range"][1] * 1.25
            )
        else:
            fa_bounds = (0.0, 1e-6)

        if has_ggbs:
            ggbs_bounds = (
                max(20.0, tcm_min * cfg["ggbs_pct_range"][0] * 0.85),
                tcm_max * cfg["ggbs_pct_range"][1] * 1.25
            )
        else:
            ggbs_bounds = (0.0, 1e-6)

        if has_metakaolin:
            mk_range = cfg["mk_pct_range"]
            mk_bounds = (
                max(10.0, tcm_min * mk_range[0] * 0.80),
                tcm_max * mk_range[1] * 1.20
            )
        else:
            mk_bounds = (0.0, 1e-6)

        bounds = [
            cement_bounds,
            fa_bounds,
            ggbs_bounds,
            mk_bounds,
            water_bounds,
            coarse_bounds,
            sand_bounds,
            admix_bounds
        ]

        def objective(x):
            cement, flyash, ggbs, metakaolin, water, coarse, sand, admix = x

            scm_content = flyash + ggbs + metakaolin
            binder = cement + scm_content

            if binder <= 0:
                return 1e9

            wbr = water / binder
            total_agg = sand + coarse
            sand_ratio = sand / total_agg if total_agg > 0 else 0.0

            binder_min, binder_max = cfg["binder_range"]
            if not (binder_min <= binder <= binder_max):
                return 1e8

            if not (tcm_min <= binder <= tcm_max):
                return 1e8

            cement_pct = cement / binder
            if not (cement_pct_min - 0.015 <= cement_pct <= cement_pct_max + 0.015):
                return 5e7 + abs(cement_pct - midpoint(cfg["cement_pct_range"])) * 1e6

            if wbr > cfg["wb_max"]:
                return 1e8

            features = build_features(
                cement, flyash, ggbs, metakaolin,
                water, coarse, sand, admix, age
            )

            if features is None:
                return 1e9

            features = features[FEATURE_ORDER].astype(float)
            pred_strength = float(model.predict(features)[0])

            lower = target_strength
            upper = target_strength * 1.05

            if pred_strength < lower:
                deficit = lower - pred_strength
                strength_penalty = 120000 + (deficit ** 2) * 2500
            elif pred_strength > upper:
                excess = pred_strength - upper
                strength_penalty = 40000 + (excess ** 2) * 2000
            else:
                mid = (lower + upper) / 2.0
                strength_penalty = abs(pred_strength - mid) * 80

            cost = (
                cement * COST["Cement"] +
                ggbs * COST["GGBS"] +
                flyash * COST["FlyAsh"] +
                metakaolin * COST["Metakaolin"] +
                water * COST["Water"] +
                coarse * COST["CoarseAggregate"] +
                sand * COST["Sand"] +
                admix * COST["Admixture"]
            )

            co2 = (
                cement * CO2["Cement"] +
                ggbs * CO2["GGBS"] +
                flyash * CO2["FlyAsh"] +
                metakaolin * CO2["Metakaolin"] +
                water * CO2["Water"] +
                coarse * CO2["CoarseAggregate"] +
                sand * CO2["Sand"] +
                admix * CO2["Admixture"]
            )

            scm_penalty = 0.0
            target_total_scm = scm_targets["target_total_scm"]
            actual_total_scm = scm_content / binder if binder > 0 else 0.0

            if target_total_scm > 0:
                scm_penalty += abs(actual_total_scm - target_total_scm) * 40000

            share_penalty = 0.0
            if scm_content > 1e-6 and target_total_scm > 0:
                fa_share = flyash / scm_content
                ggbs_share = ggbs / scm_content
                mk_share = metakaolin / scm_content

                if has_fa:
                    share_penalty += abs(fa_share - scm_targets["fa_share"]) * 18000
                if has_ggbs:
                    share_penalty += abs(ggbs_share - scm_targets["ggbs_share"]) * 18000
                if has_metakaolin:
                    share_penalty += abs(mk_share - scm_targets["mk_share"]) * 22000

            if has_fa and flyash < 25:
                share_penalty += (25 - flyash) * 1200
            if has_ggbs and ggbs < 25:
                share_penalty += (25 - ggbs) * 1200
            if has_metakaolin and metakaolin < 10:
                share_penalty += (10 - metakaolin) * 2000

            cement_penalty = abs(cement_pct - midpoint(cfg["cement_pct_range"])) * 45000

            wb_target = min(cfg["wb_max"] - 0.02, max(0.26, 0.60 - target_strength * 0.0038))
            wb_penalty = abs(wbr - wb_target) * 700

            if target_strength < 50:
                req_pct = 0.8 / 100.0
            elif target_strength < 70:
                req_pct = 1.0 / 100.0
            else:
                req_pct = 1.2 / 100.0

            min_admix = binder * req_pct
            admix_penalty = max(0.0, min_admix - admix) * 2500

            sand_ratio_target = cfg["sand_ratio_target"]
            sand_ratio_penalty = abs(sand_ratio - sand_ratio_target) * 8000

            agg_binder = total_agg / binder if binder > 0 else 99
            agg_penalty = 0.0
            if agg_binder < 2.4:
                agg_penalty += (2.4 - agg_binder) * 6000
            elif agg_binder > 4.2:
                agg_penalty += (agg_binder - 4.2) * 6000

            return (
                strength_penalty
                + 0.08 * cost
                + 0.05 * co2
                + scm_penalty
                + share_penalty
                + cement_penalty
                + wb_penalty
                + admix_penalty
                + sand_ratio_penalty
                + agg_penalty
            )

        result = differential_evolution(
            objective,
            bounds,
            strategy="best1bin",
            maxiter=320,
            popsize=22,
            tol=0.002,
            seed=42,
            polish=True,
        )

        cement, flyash, ggbs, metakaolin, water, coarse, sand, admix = result.x

        final_features = build_features(
            cement, flyash, ggbs, metakaolin,
            water, coarse, sand, admix, age
        )

        if final_features is None:
            raise Exception("Final feature generation failed.")

        final_features = final_features[FEATURE_ORDER].astype(float)
        pred_strength = float(model.predict(final_features)[0])

        pure_cost = (
            cement * COST["Cement"] +
            ggbs * COST["GGBS"] +
            flyash * COST["FlyAsh"] +
            metakaolin * COST["Metakaolin"] +
            water * COST["Water"] +
            coarse * COST["CoarseAggregate"] +
            sand * COST["Sand"] +
            admix * COST["Admixture"]
        )

        pure_co2 = (
            cement * CO2["Cement"] +
            ggbs * CO2["GGBS"] +
            flyash * CO2["FlyAsh"] +
            metakaolin * CO2["Metakaolin"] +
            water * CO2["Water"] +
            coarse * CO2["CoarseAggregate"] +
            sand * CO2["Sand"] +
            admix * CO2["Admixture"]
        )

        binder = cement + flyash + ggbs + metakaolin
        wbr_final = water / binder if binder > 0 else 0.0

        return jsonify({
            "Cement": float(round(cement, 1)),
            "FlyAsh": float(round(flyash, 1)),
            "GGBS": float(round(ggbs, 1)),
            "Metakaolin": float(round(metakaolin, 1)),
            "Water": float(round(water, 1)),
            "CoarseAggregate": float(round(coarse, 1)),
            "Sand": float(round(sand, 1)),
            "Admix": float(round(admix, 2)),
            "WBRatio": float(round(wbr_final, 3)),
            "Binder": float(round(binder, 1)),
            "cost": float(round(pure_cost, 1)),
            "co2": float(round(pure_co2, 1)),
            "pred_strength": float(round(pred_strength, 2)),
        })

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)