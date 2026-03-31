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
    "Cement_kg_m3",
    "Fly_Ash_kg_m3",
    "GGBS_kg_m3",
    "metakolin_kg_m3",
    "Water_kg_m3",
    "Sand_kg_m3",
    "AGE",
    "admixture",
    "Coarse aggregate",
    "SCMContent",
    "Binder",
    "WBRatio",
    "AggregateToBinder",
    "AdmixtureToBinder",
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
    # M30
    if target_strength < 40:
        return {
            "tcm_range": (330, 360),
            "cement_pct_range": (0.70, 0.75),
            "fa_pct_range": (0.25, 0.30),
            "ggbs_pct_range": (0.00, 0.00),
            "mk_pct_range": (0.00, 0.00),
            "water_bounds": (175, 205),
            "admix_bounds": (1.5, 5.0),
            "sand_bounds": (700, 850),
            "coarse_bounds": (1100, 1420),
            "wb_max": 0.58,
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
            "sand_bounds": (700, 850),
            "coarse_bounds": (1100, 1420),
            "wb_max": 0.55,
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
            "sand_bounds": (700, 850),
            "coarse_bounds": (1100, 1400),
            "wb_max": 0.45,
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
            "sand_bounds": (700, 850),
            "coarse_bounds": (1100, 1420),
            "wb_max": 0.40,
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
            "sand_bounds": (700, 850),
            "coarse_bounds": (1100, 1360),
            "wb_max": 0.38,
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
            "sand_bounds": (700, 850),
            "coarse_bounds": (1100, 1340),
            "wb_max": 0.35,
            "sand_ratio_target": 0.34,
        }

    # 90 to 100 MPa
    return {
        "tcm_range": (560, 590),
        "cement_pct_range": (0.44, 0.47),
        "fa_pct_range": (0.13, 0.17),
        "ggbs_pct_range": (0.26, 0.28),
        "mk_pct_range": (0.12, 0.15),
        "water_bounds": (138, 160),
        "admix_bounds": (6.5, 15.0),
        "sand_bounds": (700, 850),
        "coarse_bounds": (1100, 1320),
        "wb_max": 0.33,
        "sand_ratio_target": 0.33,
    }


def build_optimizer_bounds(cfg, has_fa, has_ggbs, has_metakaolin):
    bounds = [cfg["tcm_range"]]  # TCM always optimized

    if has_fa:
        bounds.append(cfg["fa_pct_range"])
    if has_ggbs:
        bounds.append(cfg["ggbs_pct_range"])
    if has_metakaolin:
        bounds.append(cfg["mk_pct_range"])

    bounds.extend([
        cfg["water_bounds"],
        cfg["coarse_bounds"],
        cfg["sand_bounds"],
        cfg["admix_bounds"]
    ])
    return bounds


def decode_mix_from_vector(x, has_fa, has_ggbs, has_metakaolin):
    idx = 0
    tcm = float(x[idx])
    idx += 1

    fa_pct = float(x[idx]) if has_fa else 0.0
    if has_fa:
        idx += 1

    ggbs_pct = float(x[idx]) if has_ggbs else 0.0
    if has_ggbs:
        idx += 1

    mk_pct = float(x[idx]) if has_metakaolin else 0.0
    if has_metakaolin:
        idx += 1

    water = float(x[idx])
    idx += 1
    coarse = float(x[idx])
    idx += 1
    sand = float(x[idx])
    idx += 1
    admix = float(x[idx])

    cement_pct = 1.0 - (fa_pct + ggbs_pct + mk_pct)

    cement = tcm * cement_pct
    flyash = tcm * fa_pct
    ggbs = tcm * ggbs_pct
    metakaolin = tcm * mk_pct

    return {
        "tcm": tcm,
        "cement_pct": cement_pct,
        "fa_pct": fa_pct,
        "ggbs_pct": ggbs_pct,
        "mk_pct": mk_pct,
        "cement": cement,
        "flyash": flyash,
        "ggbs": ggbs,
        "metakaolin": metakaolin,
        "water": water,
        "coarse": coarse,
        "sand": sand,
        "admix": admix,
    }


def calculate_cost(cement, flyash, ggbs, metakaolin, water, coarse, sand, admix):
    return (
        cement * COST["Cement"] +
        ggbs * COST["GGBS"] +
        flyash * COST["FlyAsh"] +
        metakaolin * COST["Metakaolin"] +
        water * COST["Water"] +
        coarse * COST["CoarseAggregate"] +
        sand * COST["Sand"] +
        admix * COST["Admixture"]
    )


def calculate_co2(cement, flyash, ggbs, metakaolin, water, coarse, sand, admix):
    return (
        cement * CO2["Cement"] +
        ggbs * CO2["GGBS"] +
        flyash * CO2["FlyAsh"] +
        metakaolin * CO2["Metakaolin"] +
        water * CO2["Water"] +
        coarse * CO2["CoarseAggregate"] +
        sand * CO2["Sand"] +
        admix * CO2["Admixture"]
    )


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
        bounds = build_optimizer_bounds(cfg, has_fa, has_ggbs, has_metakaolin)

        def objective(x):
            mix = decode_mix_from_vector(x, has_fa, has_ggbs, has_metakaolin)

            tcm = mix["tcm"]
            cement_pct = mix["cement_pct"]
            fa_pct = mix["fa_pct"]
            ggbs_pct = mix["ggbs_pct"]
            mk_pct = mix["mk_pct"]

            cement = mix["cement"]
            flyash = mix["flyash"]
            ggbs = mix["ggbs"]
            metakaolin = mix["metakaolin"]
            water = mix["water"]
            coarse = mix["coarse"]
            sand = mix["sand"]
            admix = mix["admix"]

            binder = cement + flyash + ggbs + metakaolin

            # Strict binder = TCM
            if abs(binder - tcm) > 1e-6:
                return 1e12

            if cement <= 0:
                return 1e12

            # Only OPC selected => TCM = OPC exactly
            if not has_fa and not has_ggbs and not has_metakaolin:
                if abs(cement - tcm) > 1e-6:
                    return 1e12

            # Strict SCM ranges only for selected materials
            if has_fa:
                if not (cfg["fa_pct_range"][0] <= fa_pct <= cfg["fa_pct_range"][1]):
                    return 1e12
            else:
                if abs(flyash) > 1e-8:
                    return 1e12

            if has_ggbs:
                if not (cfg["ggbs_pct_range"][0] <= ggbs_pct <= cfg["ggbs_pct_range"][1]):
                    return 1e12
            else:
                if abs(ggbs) > 1e-8:
                    return 1e12

            if has_metakaolin:
                if not (cfg["mk_pct_range"][0] <= mk_pct <= cfg["mk_pct_range"][1]):
                    return 1e12
            else:
                if abs(metakaolin) > 1e-8:
                    return 1e12

            # If all four are selected, cement must also stay within strict range
            if has_fa and has_ggbs and has_metakaolin:
                if not (cfg["cement_pct_range"][0] <= cement_pct <= cfg["cement_pct_range"][1]):
                    return 1e12

            if binder <= 0:
                return 1e12

            wbr = water / binder
            if wbr > cfg["wb_max"]:
                return 1e10 + (wbr - cfg["wb_max"]) * 1e6

            total_agg = sand + coarse
            sand_ratio = sand / total_agg if total_agg > 0 else 0.0

            features = build_features(
                cement, flyash, ggbs, metakaolin,
                water, coarse, sand, admix, age
            )

            if features is None:
                return 1e12

            pred_strength = float(model.predict(features)[0])

            lower = target_strength
            upper = target_strength * 1.05

            if pred_strength < lower:
                deficit = lower - pred_strength
                strength_penalty = 250000 + (deficit ** 2) * 12000
            elif pred_strength > upper:
                excess = pred_strength - upper
                strength_penalty = 150000 + (excess ** 2) * 8000
            else:
                mid = (lower + upper) / 2.0
                strength_penalty = abs(pred_strength - mid) * 120

            cost = calculate_cost(cement, flyash, ggbs, metakaolin, water, coarse, sand, admix)
            co2 = calculate_co2(cement, flyash, ggbs, metakaolin, water, coarse, sand, admix)

            # Minimum admixture requirement
            if target_strength < 50:
                req_pct = 0.8 / 100.0
            elif target_strength < 70:
                req_pct = 1.0 / 100.0
            else:
                req_pct = 1.2 / 100.0

            min_admix = binder * req_pct
            admix_penalty = max(0.0, min_admix - admix) * 4000

            sand_ratio_penalty = abs(sand_ratio - cfg["sand_ratio_target"]) * 9000

            agg_binder = total_agg / binder if binder > 0 else 99
            agg_penalty = 0.0
            if agg_binder < 2.4:
                agg_penalty += (2.4 - agg_binder) * 8000
            elif agg_binder > 4.2:
                agg_penalty += (agg_binder - 4.2) * 8000

            wb_target = min(cfg["wb_max"] - 0.02, max(0.26, 0.60 - target_strength * 0.0038))
            wb_penalty = abs(wbr - wb_target) * 1200

            return (
                strength_penalty
                + 0.08 * cost
                + 0.05 * co2
                + admix_penalty
                + sand_ratio_penalty
                + agg_penalty
                + wb_penalty
            )

        # Faster optimizer settings
        result = differential_evolution(
            objective,
            bounds,
            strategy="best1bin",
            maxiter=60,
            popsize=8,
            tol=0.01,
            seed=42,
            polish=False,
            updating="deferred",
            workers=1
        )

        mix = decode_mix_from_vector(result.x, has_fa, has_ggbs, has_metakaolin)

        cement = mix["cement"]
        flyash = mix["flyash"]
        ggbs = mix["ggbs"]
        metakaolin = mix["metakaolin"]
        water = mix["water"]
        coarse = mix["coarse"]
        sand = mix["sand"]
        admix = mix["admix"]
        binder = mix["tcm"]

        final_features = build_features(
            cement, flyash, ggbs, metakaolin,
            water, coarse, sand, admix, age
        )

        if final_features is None:
            raise Exception("Final feature generation failed.")

        pred_strength = float(model.predict(final_features)[0])

        pure_cost = calculate_cost(cement, flyash, ggbs, metakaolin, water, coarse, sand, admix)
        pure_co2 = calculate_co2(cement, flyash, ggbs, metakaolin, water, coarse, sand, admix)
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
            "TCM": float(round(mix["tcm"], 1)),
            "CementPct": float(round(mix["cement_pct"] * 100, 2)),
            "FlyAshPct": float(round(mix["fa_pct"] * 100, 2)),
            "GGBSPct": float(round(mix["ggbs_pct"] * 100, 2)),
            "MetakaolinPct": float(round(mix["mk_pct"] * 100, 2)),
        })

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)