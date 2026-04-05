"""Diagnostic output for the research pipeline.

Writes eval_detail and feature importance TSVs after each experiment run.
These files provide the research agent with sub-metric and feature context.
Imported by the PIPELINE section of experiment.py.
"""

import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------


def extract_importances(model, feature_cols):
    """Extract feature importances from any model.

    Walks the model and its sub-attributes to find importance-providing
    components. Works with sklearn, XGBoost, LightGBM, CatBoost, and
    arbitrary ensembles that store sub-models as instance attributes.

    Returns (extracted, skipped):
        extracted: list of (model_name, {feature: importance}) tuples
        skipped: list of model_name strings that were found but don't provide importances
    """
    extracted = []
    skipped = []
    _extract_recursive(model, feature_cols, extracted, skipped, depth=0, seen=set())
    return extracted, skipped


# Markers for objects that look like ML models (have fit/predict) but don't provide importances
_MODEL_MARKERS = ("fit", "predict", "predict_proba", "forward")


def _looks_like_model(obj):
    return any(hasattr(obj, m) and callable(getattr(obj, m)) for m in _MODEL_MARKERS)


def _extract_recursive(obj, feature_cols, extracted, skipped, depth, seen):
    if depth > 3 or id(obj) in seen:
        return
    seen.add(id(obj))

    n = len(feature_cols)
    found = False

    # sklearn standard: XGBClassifier, RF, ExtraTrees, GBM, AdaBoost, etc.
    if hasattr(obj, "feature_importances_"):
        imp = np.asarray(obj.feature_importances_)
        if len(imp) == n:
            extracted.append((type(obj).__name__, dict(zip(feature_cols, imp.astype(float)))))
            return

    # CatBoost
    if hasattr(obj, "get_feature_importance") and callable(obj.get_feature_importance):
        try:
            imp = np.asarray(obj.get_feature_importance())
            if len(imp) == n:
                extracted.append((type(obj).__name__, dict(zip(feature_cols, imp.astype(float)))))
                return
        except Exception:
            pass

    # LightGBM Booster (from lgb.train)
    if hasattr(obj, "feature_importance") and callable(obj.feature_importance):
        try:
            imp = np.asarray(obj.feature_importance(importance_type="gain"))
            if len(imp) == n:
                extracted.append((type(obj).__name__, dict(zip(feature_cols, imp.astype(float)))))
                return
        except Exception:
            pass

    # XGBoost Booster (from xgb.train)
    if hasattr(obj, "get_score") and callable(obj.get_score):
        try:
            score_dict = obj.get_score(importance_type="gain")
            mapped = {}
            for k, v in score_dict.items():
                if k.startswith("f") and k[1:].isdigit():
                    idx = int(k[1:])
                    if idx < n:
                        mapped[feature_cols[idx]] = float(v)
                elif k in feature_cols:
                    mapped[k] = float(v)
            if mapped:
                extracted.append((type(obj).__name__, mapped))
                return
        except Exception:
            pass

    # Walk instance attributes for composite/ensemble models
    obj_dict = getattr(obj, "__dict__", None)
    if obj_dict is None:
        # Leaf object with no importance and no children
        if _looks_like_model(obj):
            skipped.append(type(obj).__name__)
        return

    child_found = False
    for attr_name in sorted(obj_dict):
        if attr_name.startswith("__"):
            continue
        attr_val = obj_dict[attr_name]
        if attr_val is None or isinstance(attr_val, (int, float, str, bool, bytes, np.ndarray, dict, set, type)):
            continue
        # Recurse into lists/tuples (e.g. sklearn estimators_)
        if isinstance(attr_val, (list, tuple)):
            for item in attr_val:
                if item is not None and hasattr(item, "__dict__"):
                    _extract_recursive(item, feature_cols, extracted, skipped, depth + 1, seen)
                    child_found = True
            continue
        if hasattr(attr_val, "__dict__") or _looks_like_model(attr_val):
            _extract_recursive(attr_val, feature_cols, extracted, skipped, depth + 1, seen)
            child_found = True

    # If this object looks like a model but neither it nor its children provided importances
    if not child_found and _looks_like_model(obj):
        name = type(obj).__name__
        if name not in [e[0] for e in extracted] and name not in skipped:
            skipped.append(name)


# ---------------------------------------------------------------------------
# Feature importance TSV
# ---------------------------------------------------------------------------


def write_feat_importance(importances_by_fold, skipped_by_fold, feature_cols, path):
    """Write feature importance TSV from multiple folds.

    Args:
        importances_by_fold: list of extracted importances per fold (from extract_importances)
        skipped_by_fold: list of skipped model names per fold (from extract_importances)
        feature_cols: list of feature names
        path: output file path
    """
    if not importances_by_fold and not skipped_by_fold:
        return

    # Collect unique extracted/skipped model names across all folds
    extracted_names = sorted({name for fold in importances_by_fold for name, _ in fold})
    skipped_names = sorted({name for fold in skipped_by_fold for name in fold})

    lines = []

    # Models section — always written so agent knows what happened
    lines.append("## MODELS")
    if extracted_names:
        lines.append(f"extracted: {', '.join(extracted_names)}")
    if skipped_names:
        lines.append(f"no_importance: {', '.join(skipped_names)}")
    if not extracted_names and not skipped_names:
        lines.append("no models found")

    has_importances = any(fold for fold in importances_by_fold)
    if not has_importances:
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return

    lines.append("")

    # Collect per-feature values across folds (normalized per fold)
    feat_values = defaultdict(list)
    for fold_imps in importances_by_fold:
        fold_total = defaultdict(float)
        for _, imp_dict in fold_imps:
            for feat, val in imp_dict.items():
                fold_total[feat] += val

        s = sum(fold_total.values())
        if s > 0:
            for feat in fold_total:
                fold_total[feat] /= s

        for feat in feature_cols:
            feat_values[feat].append(fold_total.get(feat, 0.0))

    feat_stats = []
    for feat in feature_cols:
        vals = feat_values[feat]
        feat_stats.append((feat, float(np.mean(vals)), float(np.std(vals))))

    feat_stats.sort(key=lambda x: -x[1])

    lines.append("## TOP 20 BY GAIN (averaged across folds)")
    lines.append("rank\tfeature\tmean_gain\tstd_gain")
    for i, (feat, mean, std) in enumerate(feat_stats[:20], 1):
        lines.append(f"{i}\t{feat}\t{mean:.4f}\t{std:.4f}")

    lines.append("")
    lines.append("## BOTTOM 20 BY GAIN (averaged across folds)")
    lines.append("rank\tfeature\tmean_gain\tstd_gain")
    bottom = feat_stats[-20:] if len(feat_stats) >= 20 else feat_stats
    for feat, mean, std in bottom:
        rank = feat_stats.index((feat, mean, std)) + 1
        lines.append(f"{rank}\t{feat}\t{mean:.6f}\t{std:.6f}")

    lines.append("")
    lines.append("## SUMMARY")
    lines.append(f"total_features: {len(feature_cols)}")
    top20_share = sum(m for _, m, _ in feat_stats[:20])
    lines.append(f"top20_cumulative_share: {top20_share:.2f}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Eval detail TSV
# ---------------------------------------------------------------------------


def _fmt_budget(frac):
    """Format budget fraction as percentage string."""
    pct = frac * 100
    if pct < 1:
        return f"{pct:.2f}%"
    return f"{pct:.0f}%" if pct == int(pct) else f"{pct:.1f}%"


def write_eval_detail(fold_scores, fold_results, path):
    """Write eval detail TSV from fold results.

    Args:
        fold_scores: list of float scores per fold
        fold_results: list of tiered_eval result dicts per fold
        path: output file path
    """
    lines = []

    # Section 1: Fold summary
    lines.append("## FOLDS")
    lines.append("fold\tscore\tap\tauc")

    aps, aucs = [], []
    for i, (score, results) in enumerate(zip(fold_scores, fold_results), 1):
        t1 = results.get("tier1", {})
        ap = t1.get("avg_precision", 0.0)
        auc = t1.get("roc_auc", 0.0)
        aps.append(ap)
        aucs.append(auc)
        lines.append(f"{i}\t{score:.4f}\t{ap:.4f}\t{auc:.4f}")

    s = np.array(fold_scores)
    lines.append(f"mean\t{np.mean(s):.4f}\t{np.mean(aps):.4f}\t{np.mean(aucs):.4f}")
    lines.append(f"std\t{np.std(s):.4f}\t{np.std(aps):.4f}\t{np.std(aucs):.4f}")

    lines.append("")

    # Section 2: Cell detail (averaged across folds)
    budgets_set = set()
    horizons_set = set()
    for results in fold_results:
        for key, info in results.get("tier2", {}).items():
            if key == "_meta":
                continue
            budgets_set.add(key)
            for h in info.get("cells", {}):
                horizons_set.add(h)

    budgets = sorted(budgets_set)
    horizons = sorted(horizons_set)

    cell_metrics = ["weighted", "effective_n", "excess", "win_rate", "worst_decile", "knife_rate", "tail_mae", "entry_slippage"]

    lines.append("## CELLS (averaged across folds)")
    lines.append("budget\thorizon\tweighted\teff_n\texcess\twin_rate\tworst_dec\tknife\ttail_mae\tslip")

    for budget in budgets:
        for horizon in horizons:
            vals = {m: [] for m in cell_metrics}

            for results in fold_results:
                cell = results.get("tier2", {}).get(budget, {}).get("cells", {}).get(horizon)
                if cell:
                    for m in cell_metrics:
                        vals[m].append(cell[m])

            if not vals["weighted"]:
                continue

            lines.append(
                f"{_fmt_budget(budget)}\t{horizon}d\t"
                f"{np.mean(vals['weighted']):.2f}\t"
                f"{np.mean(vals['effective_n']):.1f}\t"
                f"{np.mean(vals['excess']):+.4f}\t"
                f"{np.mean(vals['win_rate']):.3f}\t"
                f"{np.mean(vals['worst_decile']):.3f}\t"
                f"{np.mean(vals['knife_rate']):.3f}\t"
                f"{np.mean(vals['tail_mae']):.3f}\t"
                f"{np.mean(vals['entry_slippage']):.3f}"
            )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
