"""Cross-fitted normative model.

For every feature: OLS  feature ~ 1 + age + age^2 + sex + site + eTIV, fitted on TRAIN subjects only.
Train subjects get out-of-fold predictions (K-fold cross-fitting over the train split, using the
cv_fold column of data/splits/ixi_final.csv), so no subject's z comes from a model that saw it.
Val/test subjects use the model fitted on all train subjects. z = residual / SD(train residuals of
the model used); percentile = 100 Phi(z); flag = |z| > 2. Subjects without age or sex get null + reason.

LEAKAGE WARNING: z-scores encode age and sex. They must never be used as inputs when age or sex is the
prediction target (E4/E5/E6); use raw or eTIV-normalised values there.
"""
from __future__ import annotations

import math

import numpy as np

SITES = ("Guys", "HH", "IOP")
FLAG_ABS_Z = 2.0
MIN_TRAIN = 50


def _design(age, sex, site, etiv, age_mean):
    a = np.asarray(age, float) - age_mean
    x = [np.ones_like(a), a, a ** 2, (np.asarray(sex) == "F").astype(float)]
    x += [(np.asarray(site) == s).astype(float) for s in SITES[1:]]
    x.append(np.asarray(etiv, float) / 1000.0)
    return np.column_stack(x)


def _fit(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    return beta, float(np.sqrt((resid ** 2).sum() / dof))


def fit_predict(table: dict[str, dict], features: list[str], train_ids, folds: dict[str, int], exclude=()) -> dict:
    """table[sid] = {"age", "sex", "site", "etiv", <feature>: value}. Returns {sid: {"z": {...}, "percentile": {...},
    "flags": [...], ...reasons}} for every subject in `table`."""
    usable = {s for s, r in table.items() if r.get("age") is not None and r.get("sex") in ("M", "F") and r.get("etiv") is not None}
    train = [s for s in train_ids if s in usable and s not in set(exclude)]
    out = {s: {"z": {}, "percentile": {}, "flags": []} for s in table}
    if len(train) < MIN_TRAIN:
        for s in table:
            out[s] = {"z": None, "percentile": None, "flags": [],
                      "z_reason": f"normative model needs >= {MIN_TRAIN} usable train subjects (have {len(train)})",
                      "percentile_reason": f"normative model needs >= {MIN_TRAIN} usable train subjects (have {len(train)})"}
        return {"subjects": out, "models": {}, "age_mean_centre": None,
                "reference": f"not fitted: {len(train)} usable train subjects < {MIN_TRAIN}"}
    age_mean = float(np.mean([table[s]["age"] for s in train]))
    for s in table:
        if s not in usable:
            out[s]["z"] = None
            out[s]["z_reason"] = "age, sex or eTIV missing (see demographics); no normative comparison"
            out[s]["percentile"] = None
            out[s]["percentile_reason"] = out[s]["z_reason"]
    def X_of(ids):
        return _design([table[i]["age"] for i in ids], [table[i]["sex"] for i in ids], [table[i]["site"] for i in ids],
                       [table[i]["etiv"] for i in ids], age_mean)
    model_info = {}
    for f in features:
        y_tr = np.array([table[s][f] for s in train], float)
        beta_all, sd_all = _fit(X_of(train), y_tr)
        model_info[f] = {"n_train": len(train), "residual_sd_all_train": sd_all}
        pred = {}
        for k in sorted({folds[s] for s in train}):
            fit_ids = [s for s in train if folds[s] != k]
            pr_ids = [s for s in train if folds[s] == k]
            b, sd = _fit(X_of(fit_ids), np.array([table[s][f] for s in fit_ids], float))
            for s, yhat in zip(pr_ids, X_of(pr_ids) @ b):
                pred[s] = (float(yhat), sd)
        others = [s for s in table if s in usable and s not in pred]
        if others:
            for s, yhat in zip(others, X_of(others) @ beta_all):
                pred[s] = (float(yhat), sd_all)
        for s, (yhat, sd) in pred.items():
            if not sd > 0:
                continue
            z = (table[s][f] - yhat) / sd
            out[s]["z"][f] = round(z, 3)
            out[s]["percentile"][f] = round(100 * 0.5 * (1 + math.erf(z / math.sqrt(2))), 2)
            if abs(z) > FLAG_ABS_Z:
                out[s]["flags"].append(f)
    return {"subjects": out, "models": model_info, "age_mean_centre": age_mean,
            "reference": f"IXI train split, n={len(train)}, OLS (age, age^2, sex, site, eTIV), 5-fold cross-fitted for train subjects"}
