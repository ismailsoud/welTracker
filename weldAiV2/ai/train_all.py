"""
WeldTracker AI - Comprehensive Model Training Pipeline.

Trains all three models with:
  - Stratified K-Fold cross-validation
  - Hyperparameter tuning via RandomizedSearchCV
  - Multi-algorithm comparison
  - Model persistence via joblib
  - Full evaluation reports (accuracy, F1, confusion matrix, ROC-AUC)
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from ai.data_generator import (
    generate_acoustic_dataset,
    generate_anomaly_dataset,
    generate_defect_dataset,
    generate_and_save_all,
)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def _load_or_generate(filename: str, gen_fn, **kwargs) -> pd.DataFrame:
    fpath = DATA_DIR / filename
    if fpath.exists():
        print(f"  Loading existing {filename}")
        return pd.read_csv(fpath)
    print(f"  Generating {filename} ...")
    df = gen_fn(**kwargs)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(fpath, index=False)
    print(f"  Saved {filename} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# 1. ANOMALY DETECTOR  (Unsupervised → Isolation Forest + One-Class SVM)
# ---------------------------------------------------------------------------

def train_anomaly_detector():
    print("\n" + "=" * 60)
    print("TRAINING: Anomaly Detector (Isolation Forest)")
    print("=" * 60)

    df = _load_or_generate("anomaly_dataset.csv", generate_anomaly_dataset, n_samples=5000)

    feature_cols = ["current_a", "voltage_v", "travel_speed_mm_min", "interpass_temp_c"]
    X = df[feature_cols].values
    y_true = df["is_anomaly"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n--- Algorithm Comparison ---")
    candidates = {
        "IsolationForest_100": IsolationForest(n_estimators=100, contamination=0.08, random_state=42),
        "IsolationForest_200": IsolationForest(n_estimators=200, contamination=0.08, random_state=42),
        "IsolationForest_300_c10": IsolationForest(n_estimators=300, contamination=0.10, random_state=42),
        "IsolationForest_200_c05": IsolationForest(n_estimators=200, contamination=0.05, random_state=42),
    }

    best_name, best_f1, best_model = None, -1, None
    results = {}
    for name, model in candidates.items():
        model.fit(X_scaled)
        preds = model.predict(X_scaled)
        preds_binary = (preds == -1).astype(int)
        f1 = f1_score(y_true, preds_binary, zero_division=0)
        results[name] = round(f1, 4)
        print(f"  {name:35s}  F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, model

    print(f"\n  Best: {best_name} (F1={best_f1:.4f})")

    normal_data = df[df["is_anomaly"] == 0][feature_cols].values
    X_normal_scaled = scaler.transform(normal_data)

    control_limits = {}
    for i, col in enumerate(feature_cols):
        vals = normal_data[:, i]
        mean, std = vals.mean(), vals.std()
        control_limits[col] = {
            "mean": round(float(mean), 4),
            "std": round(float(std), 4),
            "ucl": round(float(mean + 3 * std), 4),
            "lcl": round(float(mean - 3 * std), 4),
            "uwl": round(float(mean + 2 * std), 4),
            "lwl": round(float(mean - 2 * std), 4),
        }

    preds_binary = (best_model.predict(X_scaled) == -1).astype(int)
    report = classification_report(y_true, preds_binary, target_names=["normal", "anomaly"],
                                   output_dict=True)
    cm = confusion_matrix(y_true, preds_binary)

    artifact = {
        "model": best_model,
        "scaler": scaler,
        "control_limits": control_limits,
        "feature_cols": feature_cols,
        "metrics": {
            "best_algorithm": best_name,
            "f1_score": round(best_f1, 4),
            "precision": round(report.get("anomaly", {}).get("precision", 0), 4),
            "recall": round(report.get("anomaly", {}).get("recall", 0), 4),
            "confusion_matrix": cm.tolist(),
            "all_candidates": results,
            "n_samples": len(df),
            "anomaly_ratio": round(float(df["is_anomaly"].mean()), 4),
        },
    }

    save_path = MODEL_DIR / "anomaly_detector.joblib"
    joblib.dump(artifact, save_path)
    print(f"\n  Saved -> {save_path}")
    print(f"  Metrics: precision={artifact['metrics']['precision']:.4f}  "
          f"recall={artifact['metrics']['recall']:.4f}  f1={best_f1:.4f}")
    return artifact


# ---------------------------------------------------------------------------
# 2. DEFECT PREDICTOR  (Binary Classification)
# ---------------------------------------------------------------------------

def train_defect_predictor():
    print("\n" + "=" * 60)
    print("TRAINING: Defect Predictor (Binary Classification)")
    print("=" * 60)

    df = _load_or_generate("defect_dataset.csv", generate_defect_dataset, n_samples=5000)

    feature_cols = [
        "current_mean", "current_std", "current_min", "current_max",
        "voltage_mean", "voltage_std", "voltage_min", "voltage_max",
        "travel_speed_mean", "travel_speed_std",
        "interpass_temp_mean", "interpass_temp_max",
        "heat_input_mean", "heat_input_std",
        "current_voltage_ratio", "arc_stability_index",
        "param_deviation_score", "pass_count",
        "acoustic_energy_mean", "acoustic_amplitude_max",
        "acoustic_anomaly_ratio",
    ]
    X = df[feature_cols].values
    y = df["has_defect"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n--- Algorithm Comparison (5-Fold CV) ---")
    algo_candidates = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, class_weight="balanced", n_jobs=1),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42),
    }

    cv_results = {}
    best_algo_name, best_cv_f1 = None, -1
    for name, model in algo_candidates.items():
        scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="f1")
        mean_f1 = scores.mean()
        cv_results[name] = {"mean_f1": round(mean_f1, 4), "std_f1": round(scores.std(), 4)}
        print(f"  {name:25s}  F1={mean_f1:.4f} +/- {scores.std():.4f}")
        if mean_f1 > best_cv_f1:
            best_cv_f1, best_algo_name = mean_f1, name

    print(f"\n  Best algorithm: {best_algo_name} (CV F1={best_cv_f1:.4f})")

    print("\n--- Hyperparameter Tuning (RandomizedSearchCV) ---")
    if best_algo_name == "RandomForest":
        param_dist = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [6, 8, 10, 12, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": ["balanced", "balanced_subsample"],
        }
        base = RandomForestClassifier(random_state=42, n_jobs=1)
    else:
        param_dist = {
            "n_estimators": [150, 200, 300, 400],
            "max_depth": [4, 5, 6, 8],
            "learning_rate": [0.05, 0.1, 0.15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.8, 0.9, 1.0],
        }
        base = GradientBoostingClassifier(random_state=42)

    search = RandomizedSearchCV(
        base, param_dist, n_iter=20, cv=skf, scoring="f1",
        random_state=42, n_jobs=1, verbose=0,
    )
    search.fit(X_scaled, y)
    best_model = search.best_estimator_
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV F1:  {search.best_score_:.4f}")

    best_model.fit(X_scaled, y)

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2,
                                                random_state=42, stratify=y)
    best_model_eval = search.best_estimator_.__class__(**search.best_params_, random_state=42)
    best_model_eval.fit(X_tr, y_tr)
    y_pred = best_model_eval.predict(X_te)
    y_proba = best_model_eval.predict_proba(X_te)

    report = classification_report(y_te, y_pred, output_dict=True)
    cm = confusion_matrix(y_te, y_pred)

    try:
        roc = roc_auc_score(y_te, y_proba[:, 1])
    except Exception:
        roc = 0.0

    importances = {}
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        importances = {feature_cols[i]: round(float(imp[i]), 4) for i in range(len(feature_cols))}
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    artifact = {
        "model": best_model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "feature_importances": importances,
        "metrics": {
            "best_algorithm": best_algo_name,
            "best_params": {k: v for k, v in search.best_params_.items()},
            "cv_f1": round(search.best_score_, 4),
            "test_accuracy": round(report["accuracy"], 4),
            "test_precision": round(report.get("1", {}).get("precision", 0), 4),
            "test_recall": round(report.get("1", {}).get("recall", 0), 4),
            "test_f1": round(report.get("1", {}).get("f1-score", 0), 4),
            "roc_auc": round(roc, 4),
            "confusion_matrix": cm.tolist(),
            "cv_comparison": cv_results,
            "n_samples": len(df),
        },
    }

    save_path = MODEL_DIR / "defect_predictor.joblib"
    joblib.dump(artifact, save_path)
    print(f"\n  Saved -> {save_path}")
    print(f"  Test:  accuracy={artifact['metrics']['test_accuracy']:.4f}  "
          f"f1={artifact['metrics']['test_f1']:.4f}  "
          f"roc_auc={artifact['metrics']['roc_auc']:.4f}")
    return artifact


# ---------------------------------------------------------------------------
# 3. ACOUSTIC CLASSIFIER  (5-Class Classification)
# ---------------------------------------------------------------------------

def train_acoustic_classifier():
    print("\n" + "=" * 60)
    print("TRAINING: Acoustic Defect Classifier (5-Class)")
    print("=" * 60)

    df = _load_or_generate("acoustic_dataset.csv", generate_acoustic_dataset, n_samples=6000)

    feature_cols = sorted([
        "amplitude_db", "counts", "duration_ms", "energy", "kurtosis",
        "peak_frequency_hz", "rise_time_ms", "rms_amplitude",
        "signal_strength", "spectral_centroid_hz", "spectral_entropy",
        "spectral_spread_hz", "zero_crossing_rate",
    ])
    X = df[feature_cols].values
    y = df["defect_class_idx"].values
    class_names = ["normal", "porosity", "crack", "lack_of_fusion", "slag_inclusion"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n--- Algorithm Comparison (5-Fold CV, macro F1) ---")
    algo_candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, class_weight="balanced", n_jobs=1),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
    }

    cv_results = {}
    best_algo_name, best_cv_f1 = None, -1
    for name, model in algo_candidates.items():
        scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="f1_macro")
        mean_f1 = scores.mean()
        cv_results[name] = {"mean_f1": round(mean_f1, 4), "std_f1": round(scores.std(), 4)}
        print(f"  {name:25s}  macro-F1={mean_f1:.4f} +/- {scores.std():.4f}")
        if mean_f1 > best_cv_f1:
            best_cv_f1, best_algo_name = mean_f1, name

    print(f"\n  Best algorithm: {best_algo_name} (CV macro-F1={best_cv_f1:.4f})")

    print("\n--- Hyperparameter Tuning (RandomizedSearchCV) ---")
    if best_algo_name == "RandomForest":
        param_dist = {
            "n_estimators": [150, 200, 300, 400, 500],
            "max_depth": [8, 10, 12, 15, None],
            "min_samples_split": [2, 3, 5, 8],
            "min_samples_leaf": [1, 2, 3],
            "class_weight": ["balanced", "balanced_subsample", None],
        }
        base = RandomForestClassifier(random_state=42, n_jobs=1)
    else:
        param_dist = {
            "n_estimators": [150, 200, 300, 400],
            "max_depth": [4, 6, 8, 10],
            "learning_rate": [0.05, 0.1, 0.15],
            "min_samples_split": [2, 5, 10],
            "subsample": [0.8, 0.9, 1.0],
        }
        base = GradientBoostingClassifier(random_state=42)

    search = RandomizedSearchCV(
        base, param_dist, n_iter=20, cv=skf, scoring="f1_macro",
        random_state=42, n_jobs=1, verbose=0,
    )
    search.fit(X_scaled, y)
    best_model = search.best_estimator_
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV macro-F1: {search.best_score_:.4f}")

    best_model.fit(X_scaled, y)

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2,
                                                random_state=42, stratify=y)
    eval_model = search.best_estimator_.__class__(**search.best_params_,
                                                    random_state=42 if hasattr(search.best_estimator_, "random_state") else None)
    if hasattr(eval_model, "random_state"):
        eval_model.random_state = 42
    eval_model.fit(X_tr, y_tr)
    y_pred = eval_model.predict(X_te)

    report = classification_report(y_te, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_te, y_pred)

    importances = {}
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        importances = {feature_cols[i]: round(float(imp[i]), 4) for i in range(len(feature_cols))}
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    per_class = {}
    for cls_name in class_names:
        if cls_name in report:
            per_class[cls_name] = {
                "precision": round(report[cls_name]["precision"], 4),
                "recall": round(report[cls_name]["recall"], 4),
                "f1": round(report[cls_name]["f1-score"], 4),
                "support": int(report[cls_name]["support"]),
            }

    artifact = {
        "model": best_model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "class_names": class_names,
        "feature_importances": importances,
        "metrics": {
            "best_algorithm": best_algo_name,
            "best_params": {k: (v if not isinstance(v, np.integer) else int(v))
                           for k, v in search.best_params_.items()},
            "cv_macro_f1": round(search.best_score_, 4),
            "test_accuracy": round(report["accuracy"], 4),
            "test_macro_f1": round(report.get("macro avg", {}).get("f1-score", 0), 4),
            "test_weighted_f1": round(report.get("weighted avg", {}).get("f1-score", 0), 4),
            "confusion_matrix": cm.tolist(),
            "per_class_metrics": per_class,
            "cv_comparison": cv_results,
            "n_samples": len(df),
        },
    }

    save_path = MODEL_DIR / "acoustic_classifier.joblib"
    joblib.dump(artifact, save_path)
    print(f"\n  Saved -> {save_path}")
    print(f"  Test:  accuracy={artifact['metrics']['test_accuracy']:.4f}  "
          f"macro-F1={artifact['metrics']['test_macro_f1']:.4f}")
    for cls_name, m in per_class.items():
        print(f"    {cls_name:18s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    return artifact


# ---------------------------------------------------------------------------
# Save training summary report
# ---------------------------------------------------------------------------

def save_training_report(anomaly_art, defect_art, acoustic_art):
    report = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "anomaly_detector": anomaly_art["metrics"],
        "defect_predictor": defect_art["metrics"],
        "acoustic_classifier": acoustic_art["metrics"],
    }
    report_path = MODEL_DIR / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Training report saved -> {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_all():
    print("=" * 60)
    print("WeldTracker AI - Full Model Training Pipeline")
    print("=" * 60)
    t0 = time.time()

    print("\n[1/4] Generating / loading datasets ...")
    generate_and_save_all(overwrite=False)

    print("\n[2/4] Training anomaly detector ...")
    anomaly_art = train_anomaly_detector()

    print("\n[3/4] Training defect predictor ...")
    defect_art = train_defect_predictor()

    print("\n[4/4] Training acoustic classifier ...")
    acoustic_art = train_acoustic_classifier()

    save_training_report(anomaly_art, defect_art, acoustic_art)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"ALL MODELS TRAINED SUCCESSFULLY in {elapsed:.1f}s")
    print("=" * 60)
    print(f"\nArtifacts saved in: {MODEL_DIR}")
    print("  - anomaly_detector.joblib")
    print("  - defect_predictor.joblib")
    print("  - acoustic_classifier.joblib")
    print("  - training_report.json")

    return {
        "anomaly_detector": anomaly_art,
        "defect_predictor": defect_art,
        "acoustic_classifier": acoustic_art,
    }


if __name__ == "__main__":
    train_all()
