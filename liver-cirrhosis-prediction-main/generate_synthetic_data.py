"""
Realistic Synthetic Data Generator for Liver Cirrhosis Dataset
───────────────────────────────────────────────────────────────
Generates synthetic patients using:
  - Per-stage clinical distributions derived from cirrhosis.csv
  - Correlated lab values (bilirubin↑ → albumin↓, etc.)
  - Realistic clinical constraints and stage-appropriate symptom prevalence
  - Gaussian copula-style correlation to mirror real patient co-variation

Output: datasets/cirrhosis_augmented.csv  (real + synthetic, ~5000 patients)
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(seed=42)

# ────────────────────────────────────────────────────────────────────────────
# Stage-specific clinical parameters (derived from real data analysis above)
# All values calibrated to match observed mean ± std per stage
# ────────────────────────────────────────────────────────────────────────────
STAGE_PARAMS = {
    1: dict(
        n_synth      = 1200,          # target synthetic count per stage
        age_mean     = 47,   age_std  = 10,   age_min = 25, age_max = 70,
        bili_shape   = 1.2,  bili_scale = 0.5,               # gamma dist, mild elevation
        albumin_mean = 3.72, albumin_std = 0.30,
        sgot_mean    = 90,   sgot_std   = 38,
        alkp_shape   = 1.5,  alkp_scale = 600,
        plat_mean    = 290,  plat_std   = 80,
        pt_mean      = 10.8, pt_std     = 0.8,
        chol_mean    = 284,  chol_std   = 90,
        copper_mean  = 65,   copper_std = 38,
        trig_mean    = 97,   trig_std   = 28,
        female_p     = 0.86,
        ascites_p    = 0.00,
        hepato_p     = 0.05,
        spiders_p    = 0.05,
        edema_p      = [0.95, 0.04, 0.01],   # N, S, Y
    ),
    2: dict(
        n_synth      = 1200,
        age_mean     = 49,   age_std  = 10,   age_min = 28, age_max = 76,
        bili_shape   = 1.4,  bili_scale = 1.2,
        albumin_mean = 3.60, albumin_std = 0.35,
        sgot_mean    = 115,  sgot_std   = 50,
        alkp_shape   = 1.5,  alkp_scale = 800,
        plat_mean    = 284,  plat_std   = 95,
        pt_mean      = 10.5, pt_std     = 0.9,
        chol_mean    = 338,  chol_std   = 145,
        copper_mean  = 70,   copper_std = 44,
        trig_mean    = 112,  trig_std   = 43,
        female_p     = 0.91,
        ascites_p    = 0.02,
        hepato_p     = 0.21,
        spiders_p    = 0.10,
        edema_p      = [0.93, 0.05, 0.02],
    ),
    3: dict(
        n_synth      = 1200,
        age_mean     = 49,   age_std  = 10,   age_min = 25, age_max = 73,
        bili_shape   = 1.6,  bili_scale = 1.5,
        albumin_mean = 3.58, albumin_std = 0.36,
        sgot_mean    = 122,  sgot_std   = 47,
        alkp_shape   = 1.7,  alkp_scale = 900,
        plat_mean    = 263,  plat_std   = 82,
        pt_mean      = 10.5, pt_std     = 0.75,
        chol_mean    = 388,  chol_std   = 250,
        copper_mean  = 88,   copper_std = 63,
        trig_mean    = 125,  trig_std   = 50,
        female_p     = 0.90,
        ascites_p    = 0.01,
        hepato_p     = 0.34,
        spiders_p    = 0.19,
        edema_p      = [0.89, 0.09, 0.02],
    ),
    4: dict(
        n_synth      = 1200,
        age_mean     = 54,   age_std  = 11,   age_min = 28, age_max = 80,
        bili_shape   = 2.0,  bili_scale = 2.0,   # heavier tail — severe liver dysfunction
        albumin_mean = 3.28, albumin_std = 0.42,
        sgot_mean    = 127,  sgot_std   = 47,
        alkp_shape   = 1.8,  alkp_scale = 850,
        plat_mean    = 225,  plat_std   = 95,
        pt_mean      = 11.1, pt_std     = 0.95,
        chol_mean    = 328,  chol_std   = 138,
        copper_mean  = 114,  copper_std = 90,
        trig_mean    = 122,  trig_std   = 63,
        female_p     = 0.88,
        ascites_p    = 0.15,
        hepato_p     = 0.61,
        spiders_p    = 0.35,
        edema_p      = [0.72, 0.17, 0.11],
    ),
}

# Biological constraints (hard limits from medical literature)
CONSTRAINTS = {
    "Bilirubin":     (0.3,  28.0),
    "Albumin":       (2.0,   4.7),
    "SGOT":          (26,   460),
    "Alk_Phos":      (289, 14000),
    "Platelets":     (62,   720),
    "Prothrombin":   (9.0,  18.0),
    "Cholesterol":   (120, 1800),
    "Copper":        (4,    590),
    "Tryglicerides": (33,   600),
}


def _clip(arr, col):
    lo, hi = CONSTRAINTS[col]
    return np.clip(arr, lo, hi)


def _add_bilirubin_correlation(bilirubin, albumin, sgot, pt, plat):
    """High bilirubin correlates with low albumin, high PT, low platelets."""
    bili_norm = np.clip((bilirubin - 1) / 10, 0, 1)   # 0-1 severity proxy
    albumin   = albumin   - 0.25 * bili_norm + RNG.normal(0, 0.03, len(bilirubin))
    pt        = pt        + 0.40 * bili_norm + RNG.normal(0, 0.05, len(bilirubin))
    plat      = plat      - 30   * bili_norm + RNG.normal(0, 3,    len(bilirubin))
    sgot      = sgot      + 20   * bili_norm + RNG.normal(0, 2,    len(bilirubin))
    return albumin, sgot, pt, plat


def generate_stage(stage, params, n):
    p = params

    # Age (years → days to match original dataset format)
    age_yrs = RNG.normal(p["age_mean"], p["age_std"], n)
    age_yrs = np.clip(age_yrs, p["age_min"], p["age_max"])
    age_days = (age_yrs * 365.25).astype(int)

    # Lab values — gamma gives right-skewed distributions like real labs
    bilirubin   = RNG.gamma(p["bili_shape"], p["bili_scale"], n)
    albumin     = RNG.normal(p["albumin_mean"], p["albumin_std"], n)
    sgot        = RNG.normal(p["sgot_mean"], p["sgot_std"], n)
    alk_phos    = RNG.gamma(p["alkp_shape"], p["alkp_scale"], n)
    platelets   = RNG.normal(p["plat_mean"], p["plat_std"], n)
    prothrombin = RNG.normal(p["pt_mean"], p["pt_std"], n)
    cholesterol = RNG.normal(p["chol_mean"], p["chol_std"], n)
    copper      = RNG.gamma(2.0, p["copper_mean"] / 2, n)
    triglyc     = RNG.normal(p["trig_mean"], p["trig_std"], n)

    # Apply biological correlation between bilirubin and other values
    albumin, sgot, prothrombin, platelets = _add_bilirubin_correlation(
        bilirubin, albumin, sgot, prothrombin, platelets
    )

    # Clip to physiological bounds
    bilirubin   = _clip(bilirubin,   "Bilirubin")
    albumin     = _clip(albumin,     "Albumin")
    sgot        = _clip(sgot,        "SGOT")
    alk_phos    = _clip(alk_phos,    "Alk_Phos")
    platelets   = _clip(platelets,   "Platelets")
    prothrombin = _clip(prothrombin, "Prothrombin")
    cholesterol = _clip(cholesterol, "Cholesterol")
    copper      = _clip(copper,      "Copper")
    triglyc     = _clip(triglyc,     "Tryglicerides")

    # Categorical variables
    sex         = RNG.choice(["F", "M"], n, p=[p["female_p"], 1 - p["female_p"]])
    ascites     = RNG.choice(["N", "Y"], n, p=[1-p["ascites_p"], p["ascites_p"]])
    hepatomeg   = RNG.choice(["N", "Y"], n, p=[1-p["hepato_p"],  p["hepato_p"]])
    spiders     = RNG.choice(["N", "Y"], n, p=[1-p["spiders_p"], p["spiders_p"]])
    edema       = RNG.choice(["N", "S", "Y"], n, p=p["edema_p"])

    # Drug assignment (same as original trial)
    drug = RNG.choice(["D-penicillamine", "Placebo"], n, p=[0.5, 0.5])

    # N_Days — survival days (rough clinical range per stage)
    ndays_range = {1: (3000, 5000), 2: (2000, 4500), 3: (1000, 3500), 4: (200, 2500)}
    lo, hi = ndays_range[stage]
    n_days = RNG.integers(lo, hi, n)

    # Status (C=censored, D=dead) — stage 4 has higher death rate
    death_p = {1: 0.08, 2: 0.18, 3: 0.35, 4: 0.58}[stage]
    status  = RNG.choice(["C", "D"], n, p=[1 - death_p, death_p])

    df = pd.DataFrame({
        "ID":           np.arange(n) + 1,
        "N_Days":       n_days,
        "Status":       status,
        "Drug":         drug,
        "Age":          age_days,
        "Sex":          sex,
        "Ascites":      ascites,
        "Hepatomegaly": hepatomeg,
        "Spiders":      spiders,
        "Edema":        edema,
        "Bilirubin":    np.round(bilirubin, 2),
        "Cholesterol":  np.round(cholesterol, 1),
        "Albumin":      np.round(albumin, 2),
        "Copper":       np.round(copper, 1),
        "Alk_Phos":     np.round(alk_phos, 2),
        "SGOT":         np.round(sgot, 2),
        "Tryglicerides":np.round(triglyc, 1),
        "Platelets":    np.round(platelets, 1),
        "Prothrombin":  np.round(prothrombin, 1),
        "Stage":        float(stage),
        "_synthetic":   True,
    })
    return df


def main():
    print("Loading real data …")
    real = pd.read_csv("datasets/cirrhosis.csv")
    real["_synthetic"] = False
    print(f"  Real patients: {len(real)}  |  Stage dist: {real['Stage'].value_counts().sort_index().to_dict()}")

    print("\nGenerating synthetic patients …")
    synth_frames = []
    for stage, params in STAGE_PARAMS.items():
        n = params["n_synth"]
        df_s = generate_stage(stage, params, n)
        synth_frames.append(df_s)
        print(f"  Stage {stage}: {n} synthetic patients generated")

    synth_all = pd.concat(synth_frames, ignore_index=True)
    synth_all["ID"] = np.arange(len(real) + 1, len(real) + 1 + len(synth_all))

    # Combine real + synthetic
    combined = pd.concat([real, synth_all], ignore_index=True)
    combined = combined.drop(columns=["_synthetic"], errors="ignore")

    # Final distribution
    print(f"\nCombined dataset: {len(combined)} patients")
    print("Stage distribution:")
    print(combined["Stage"].value_counts().sort_index())

    out_path = Path("datasets/cirrhosis_augmented.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

    # Validation: compare real vs synthetic lab stats
    print("\n=== Validation: Real vs Synthetic means (Stage 4) ===")
    real4 = real[real["Stage"] == 4.0]
    synt4 = synth_all[synth_all["Stage"] == 4.0]
    for col in ["Bilirubin", "Albumin", "SGOT", "Platelets", "Prothrombin"]:
        r_mean = real4[col].mean()
        s_mean = synt4[col].mean()
        diff   = abs(r_mean - s_mean) / (r_mean + 1e-9) * 100
        status = "✓" if diff < 15 else "⚠"
        print(f"  {col:14s}  real={r_mean:.2f}  synth={s_mean:.2f}  diff={diff:.1f}%  {status}")


if __name__ == "__main__":
    main()
