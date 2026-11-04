"""
CS/SE 4AL3 Group 29 - Early Prediction of Student Dropout and Academic Success
Progress report pipeline.

Run:  python dropout_pipeline.py
Needs: data.csv (UCI dataset 697, semicolon-separated) in the same folder.
Outputs (in ./outputs): results.csv, per_class_test.csv, figures *.png, best_params.json
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
    classification_report, f1_score, log_loss, precision_recall_fscore_support,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
DATA_PATH = "data.csv"
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

CLASSES = ["Dropout", "Enrolled", "Graduate"]      # encoded 0, 1, 2
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}

# ----------------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, sep=";")
if df.shape[1] == 1:                                # fallback if comma-separated
    df = pd.read_csv(DATA_PATH, sep=",")
df.columns = df.columns.str.strip()                 # some headers contain a stray tab
print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

y = df["Target"].map(LABEL_MAP).values
print("\nClass counts:\n", df["Target"].value_counts())
print("Class proportions:\n", df["Target"].value_counts(normalize=True).round(3))

# Figure: class distribution
plt.figure(figsize=(4.5, 3))
df["Target"].value_counts().reindex(CLASSES).plot(kind="bar", color=["#c0392b", "#f39c12", "#27ae60"])
plt.ylabel("Students")
plt.title("Class distribution")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT / "class_distribution.png", dpi=200)
plt.close()

# ----------------------------------------------------------------------------
# 2. STAGE-WISE FEATURE SETS (leakage-safe, matches the proposal)
# ----------------------------------------------------------------------------
DROP = ["Target", "Nacionality"]                    # UCI header spells it "Nacionality"
STAGE2_EXTRA = ["Tuition fees up to date", "Debtor"]

all_feats = [c for c in df.columns if c not in DROP]
sem1 = [c for c in all_feats if "1st sem" in c]
sem2 = [c for c in all_feats if "2nd sem" in c]

stage1 = [c for c in all_feats if c not in sem1 + sem2 + STAGE2_EXTRA]
stage2 = stage1 + STAGE2_EXTRA + sem1
stage3 = stage2 + sem2
STAGES = {"Stage 1 (enrollment)": stage1,
          "Stage 2 (+sem 1)": stage2,
          "Stage 3 (+sem 2)": stage3}
for k, v in STAGES.items():
    print(f"{k}: {len(v)} features")
assert len(stage3) == len(all_feats)

# Nominal codes -> treated as categorical (integer codes have no numeric meaning)
CAT_COLS = ["Marital status", "Application mode", "Course", "Previous qualification",
            "Mother's qualification", "Father's qualification",
            "Mother's occupation", "Father's occupation"]


def make_preprocessor(cols, scale):
    cat = [c for c in CAT_COLS if c in cols]
    num = [c for c in cols if c not in cat]
    if scale:   # logistic regression: one-hot + standardize
        return ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", StandardScaler(), num)])
    return ColumnTransformer([   # tree models: ordinal codes, no scaling needed
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
        ("num", "passthrough", num)])


# ----------------------------------------------------------------------------
# 3. SPLIT: ONE stratified train/test split, shared by all stages and models
#    Train (80%) is used with 5-fold stratified CV for tuning and validation.
#    Test (20%) is touched only once per final model.
# ----------------------------------------------------------------------------
idx_train, idx_test = train_test_split(
    np.arange(len(df)), test_size=0.20, stratify=y, random_state=SEED)
y_train, y_test = y[idx_train], y[idx_test]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print(f"\nTrain: {len(idx_train)}  Test: {len(idx_test)}")

# ----------------------------------------------------------------------------
# 4. MODELS
#    class_weight='balanced' -> w_c = N / (K * N_c): rare classes cost more.
#    Loss: LR  = multinomial cross-entropy + L2 (lbfgs)
#          RF  = Gini impurity splits (trees averaged)
#          HGB = multiclass cross-entropy, boosted by gradient descent in function space
# ----------------------------------------------------------------------------
MODELS = {
    "Majority": dict(
        est=DummyClassifier(strategy="most_frequent"), scale=False, params=None),
    "LogReg": dict(
        est=LogisticRegression(max_iter=3000, class_weight="balanced"), scale=True,
        params={"clf__C": loguniform(1e-3, 1e2)}, n_iter=15),
    "RandomForest": dict(
        est=RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
        scale=False,
        params={"clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 5, 10, 20],
                "clf__min_samples_leaf": [1, 2, 5, 10],
                "clf__max_features": ["sqrt", 0.3]}, n_iter=20),
    "GradBoost": dict(
        est=HistGradientBoostingClassifier(class_weight="balanced", early_stopping=True,
                                           validation_fraction=0.1, random_state=SEED),
        scale=False,
        params={"clf__learning_rate": loguniform(0.01, 0.3),
                "clf__max_depth": [3, 5, None],
                "clf__max_leaf_nodes": [15, 31],
                "clf__min_samples_leaf": [10, 20, 40],
                "clf__l2_regularization": loguniform(1e-3, 10),
                "clf__max_iter": [200, 400]}, n_iter=20),
}


def evaluate(y_true, proba):
    pred = proba.argmax(axis=1)
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, labels=[0, 1, 2], zero_division=0)
    onehot = np.eye(3)[y_true]
    out = {
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
        "log_loss": log_loss(y_true, np.clip(proba, 1e-7, 1), labels=[0, 1, 2]),
        "brier": np.mean(np.sum((proba - onehot) ** 2, axis=1)),
    }
    for i, c in enumerate(CLASSES):
        out[f"{c}_precision"], out[f"{c}_recall"], out[f"{c}_f1"] = p[i], r[i], f[i]
    return out


# ----------------------------------------------------------------------------
# 5. TRAIN / TUNE / EVALUATE every (stage, model)
# ----------------------------------------------------------------------------
rows, best_params, fitted = [], {}, {}
for stage, cols in STAGES.items():
    X_train = df.iloc[idx_train][cols]
    X_test = df.iloc[idx_test][cols]
    for name, m in MODELS.items():
        pipe = Pipeline([("prep", make_preprocessor(cols, m["scale"])), ("clf", clone(m["est"]))])
        if m["params"]:
            search = RandomizedSearchCV(
                pipe, m["params"], n_iter=m["n_iter"], scoring="f1_macro",
                cv=cv, random_state=SEED, n_jobs=-1, refit=True)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params[f"{stage} | {name}"] = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                                                for k, v in search.best_params_.items()}
        else:
            model = pipe.fit(X_train, y_train)

        # Validation metrics: out-of-fold predictions on the training set
        oof = cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba")
        val = evaluate(y_train, oof)
        # Test metrics: model refit on the full training set
        test = evaluate(y_test, model.predict_proba(X_test))

        row = {"stage": stage, "model": name}
        row.update({f"val_{k}": v for k, v in val.items()})
        row.update({f"test_{k}": v for k, v in test.items()})
        rows.append(row)
        fitted[(stage, name)] = (model, X_test)
        print(f"{stage:22s} {name:13s} val macroF1={val['macro_f1']:.3f} "
              f"test macroF1={test['macro_f1']:.3f} test dropout recall={test['Dropout_recall']:.3f}")

res = pd.DataFrame(rows)
res.to_csv(OUT / "results.csv", index=False)
json.dump(best_params, open(OUT / "best_params.json", "w"), indent=2, default=str)

# Compact table for the report
cols_report = ["stage", "model", "test_macro_f1", "test_balanced_acc", "test_accuracy",
               "test_Dropout_recall", "test_Dropout_precision", "test_log_loss"]
print("\n", res[cols_report].round(3).to_string(index=False))

# ----------------------------------------------------------------------------
# 6. FIGURES
# ----------------------------------------------------------------------------
# (a) Stage-by-stage macro F1 and dropout recall per model
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, metric, title in zip(axes, ["test_macro_f1", "test_Dropout_recall"],
                             ["Test macro F1", "Test dropout recall"]):
    pv = res.pivot(index="stage", columns="model", values=metric)[list(MODELS)]
    pv.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(OUT / "stage_comparison.png", dpi=200)
plt.close()

# (b) Normalized confusion matrices on test for the tuned GradBoost at each stage
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, stage in zip(axes, STAGES):
    model, X_te = fitted[(stage, "GradBoost")]
    ConfusionMatrixDisplay.from_predictions(
        y_test, model.predict(X_te), display_labels=CLASSES,
        normalize="true", ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(stage, fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "confusion_gradboost.png", dpi=200)
plt.close()

# (c) Per-class report for the best model at each stage (chosen by validation macro F1)
with open(OUT / "classification_reports.txt", "w") as f:
    for stage in STAGES:
        sub = res[(res.stage == stage) & (res.model != "Majority")]
        best = sub.sort_values("val_macro_f1", ascending=False).iloc[0]["model"]
        model, X_te = fitted[(stage, best)]
        f.write(f"=== {stage} | best by validation macro F1: {best} ===\n")
        f.write(classification_report(y_test, model.predict(X_te), target_names=CLASSES, digits=3))
        f.write("\n")

# (d) Permutation importance, Stage 3, best model by validation macro F1
sub = res[(res.stage == "Stage 3 (+sem 2)") & (res.model != "Majority")]
best3 = sub.sort_values("val_macro_f1", ascending=False).iloc[0]["model"]
model, X_te = fitted[("Stage 3 (+sem 2)", best3)]
pi = permutation_importance(model, X_te, y_test, scoring="f1_macro",
                            n_repeats=10, random_state=SEED, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values().tail(15)
plt.figure(figsize=(6, 4.5))
imp.plot(kind="barh")
plt.xlabel("Drop in test macro F1 when shuffled")
plt.title(f"Permutation importance, Stage 3 ({best3})")
plt.tight_layout()
plt.savefig(OUT / "permutation_importance_stage3.png", dpi=200)
plt.close()

print("\nDone. See ./outputs/")
