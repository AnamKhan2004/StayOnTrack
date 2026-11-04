"""
CS/SE 4AL3 Group 29 - Early Prediction of Student Dropout and Academic Success
Progress report pipeline (heavily commented version).

WHAT THIS SCRIPT DOES, IN ONE PARAGRAPH
---------------------------------------
It loads the UCI student dropout dataset, builds three feature sets (one per
prediction stage: enrollment, after semester 1, after semester 2), and trains
four models on each (majority-class baseline, logistic regression, random
forest, gradient boosting). It tunes the hyperparameters with cross-validation
on the training data, then reports both validation scores (from
cross-validation) and test scores (from a held-out test set). Finally it
saves tables and figures you can paste into your report.

Run:  python dropout_pipeline.py
Needs: data.csv (UCI dataset 697, semicolon-separated) in the same folder.
Outputs (in ./outputs): results.csv, figures *.png, best_params.json,
                        classification_reports.txt
"""

# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import json                      # to save the best hyperparameters to a file
import warnings                  # to silence noisy (harmless) warnings
from pathlib import Path         # cleaner way to handle folder/file paths

import matplotlib
matplotlib.use("Agg")            # draw figures to files without opening windows
import matplotlib.pyplot as plt  # plotting
import numpy as np               # numerical arrays
import pandas as pd              # tables (DataFrames)
from scipy.stats import loguniform   # distribution for sampling hyperparameters on a log scale

from sklearn.base import clone   # makes a fresh, unfitted copy of a model
from sklearn.compose import ColumnTransformer  # apply different preprocessing to different columns
from sklearn.dummy import DummyClassifier      # the "always predict the majority class" baseline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance  # measures which features matter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,      # draws confusion matrices
    accuracy_score, balanced_accuracy_score,
    classification_report,       # text table of per-class precision/recall/F1
    f1_score, log_loss, precision_recall_fscore_support,
)
from sklearn.model_selection import (
    RandomizedSearchCV,          # tries random hyperparameter combinations using CV
    StratifiedKFold,             # CV that keeps class proportions in each fold
    cross_val_predict,           # gets out-of-fold predictions for every training row
    train_test_split,            # splits data into train and test
)
from sklearn.pipeline import Pipeline  # chains preprocessing + model into one object
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")  # hide convergence/future warnings so output stays readable

# ----------------------------------------------------------------------------
# GLOBAL SETTINGS
# ----------------------------------------------------------------------------
SEED = 42                        # fixed random seed so results are reproducible
DATA_PATH = "data.csv"           # location of the dataset
OUT = Path("outputs")            # folder where results and figures are saved
OUT.mkdir(exist_ok=True)         # create the folder if it doesn't exist

CLASSES = ["Dropout", "Enrolled", "Graduate"]         # class names, in a fixed order
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}     # name -> number: Dropout=0, Enrolled=1, Graduate=2
# Models work with numbers, so we convert the text labels to 0/1/2.

# ----------------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, sep=";")      # the UCI file uses semicolons as separators
if df.shape[1] == 1:                      # if everything landed in one column, the file used commas instead
    df = pd.read_csv(DATA_PATH, sep=",")
df.columns = df.columns.str.strip()       # remove stray whitespace/tabs in some column names
print("Shape:", df.shape)                                       # (rows, columns): expect (4424, 37)
print("Missing values:", int(df.isna().sum().sum()))            # total empty cells (should be 0)
print("Duplicate rows:", int(df.duplicated().sum()))            # fully duplicated rows

y = df["Target"].map(LABEL_MAP).values    # the label column as numbers 0/1/2
print("\nClass counts:\n", df["Target"].value_counts())                       # how many students per class
print("Class proportions:\n", df["Target"].value_counts(normalize=True).round(3))  # same, as fractions

# Figure: bar chart of class distribution (shows the class imbalance)
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
# DROP: columns we never use as inputs.
#   "Target" is the label itself (using it as an input would be cheating).
#   "Nacionality" is excluded for fairness reasons (as stated in the proposal).
#   Note the dataset's own spelling of "Nacionality".
DROP = ["Target", "Nacionality"]

# These two are placed in Stage 2 (not Stage 1) because payment status can
# change after enrollment and may reflect the student's progress or decision to leave.
STAGE2_EXTRA = ["Tuition fees up to date", "Debtor"]

all_feats = [c for c in df.columns if c not in DROP]            # every usable input column
sem1 = [c for c in all_feats if "1st sem" in c]                 # all first-semester columns
sem2 = [c for c in all_feats if "2nd sem" in c]                 # all second-semester columns

stage1 = [c for c in all_feats if c not in sem1 + sem2 + STAGE2_EXTRA]  # info known at enrollment
stage2 = stage1 + STAGE2_EXTRA + sem1                                    # + payment status + semester 1
stage3 = stage2 + sem2                                                   # + semester 2
STAGES = {"Stage 1 (enrollment)": stage1,
          "Stage 2 (+sem 1)": stage2,
          "Stage 3 (+sem 2)": stage3}
for k, v in STAGES.items():
    print(f"{k}: {len(v)} features")
assert len(stage3) == len(all_feats)   # sanity check: Stage 3 should contain every usable feature

# Some columns are integer CODES for categories (e.g. Course = 9254), not quantities.
# A model should not treat code 9254 as "bigger" than 171, so we encode them as categories.
CAT_COLS = ["Marital status", "Application mode", "Course", "Previous qualification",
            "Mother's qualification", "Father's qualification",
            "Mother's occupation", "Father's occupation"]


def make_preprocessor(cols, scale):
    """Build the preprocessing step for a given list of columns.

    scale=True  -> for logistic regression: one-hot encode categories and
                   standardize numeric columns (mean 0, std 1). Logistic
                   regression is sensitive to feature scale and to fake ordering.
    scale=False -> for tree models: ordinal-encode categories (each code becomes
                   an integer 0..k-1) and leave numbers unscaled. Trees split on
                   thresholds, so scaling doesn't matter for them.
    """
    cat = [c for c in CAT_COLS if c in cols]       # categorical columns present in this stage
    num = [c for c in cols if c not in cat]        # everything else is numeric
    if scale:
        return ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),  # one 0/1 column per category value
            ("num", StandardScaler(), num)])                        # rescale numbers
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
        ("num", "passthrough", num)])                               # numbers pass through unchanged
    # handle_unknown: if a category appears in test/validation but not in training,
    # don't crash (one-hot: all zeros; ordinal: -1).


# ----------------------------------------------------------------------------
# 3. SPLIT: ONE stratified train/test split, shared by all stages and models
#    Train (80%) is used with 5-fold stratified CV for tuning and validation.
#    Test (20%) is touched only to report final scores.
# ----------------------------------------------------------------------------
# We split row INDICES (not the data) so every stage and model uses exactly the
# same students in train and test, which makes comparisons fair.
# stratify=y keeps the class proportions the same in train and test.
idx_train, idx_test = train_test_split(
    np.arange(len(df)), test_size=0.20, stratify=y, random_state=SEED)
y_train, y_test = y[idx_train], y[idx_test]

# 5-fold stratified cross-validation: the training set is cut into 5 parts;
# each part takes a turn as validation while the other 4 train the model.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print(f"\nTrain: {len(idx_train)}  Test: {len(idx_test)}")

# ----------------------------------------------------------------------------
# 4. MODELS
#    class_weight='balanced' -> w_c = N / (K * N_c): rare classes cost more.
#       (N = total samples, K = number of classes, N_c = samples in class c.)
#       This makes mistakes on smaller classes count more during training.
#    Loss: LR  = multinomial cross-entropy + L2 (lbfgs)
#          RF  = Gini impurity splits (trees averaged)
#          HGB = multiclass cross-entropy, boosted by gradient descent in function space
# ----------------------------------------------------------------------------
# Each entry holds:
#   est    : the model object
#   scale  : which preprocessing to use (see make_preprocessor)
#   params : hyperparameter search space (None = nothing to tune)
#   n_iter : how many random combinations to try
# Parameter names start with "clf__" because the model sits inside a Pipeline
# step named "clf".
MODELS = {
    # Baseline: always predicts the most common class. Any real model must beat this.
    "Majority": dict(
        est=DummyClassifier(strategy="most_frequent"), scale=False, params=None),

    # Logistic regression: interpretable linear baseline.
    # C is the inverse regularization strength (smaller C = stronger regularization).
    # loguniform samples C across several orders of magnitude evenly.
    "LogReg": dict(
        est=LogisticRegression(max_iter=3000, class_weight="balanced"), scale=True,
        params={"clf__C": loguniform(1e-3, 1e2)}, n_iter=15),

    # Random forest: many decision trees trained on random subsets, votes averaged.
    #   n_estimators     : number of trees
    #   max_depth        : how deep each tree can grow (None = unlimited)
    #   min_samples_leaf : minimum samples in a leaf (bigger = smoother, less overfit)
    #   max_features     : how many features each split may consider
    "RandomForest": dict(
        est=RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
        scale=False,
        params={"clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 5, 10, 20],
                "clf__min_samples_leaf": [1, 2, 5, 10],
                "clf__max_features": ["sqrt", 0.3]}, n_iter=20),

    # Gradient boosting: trees built one after another, each correcting the
    # errors of the previous ones.
    #   early_stopping   : stop adding trees when a 10% internal validation split stops improving
    #   learning_rate    : step size per tree (smaller = slower but often better)
    #   max_depth / max_leaf_nodes / min_samples_leaf : tree complexity controls
    #   l2_regularization: penalty on leaf values to reduce overfitting
    #   max_iter         : maximum number of trees
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
    """Compute all metrics from true labels and predicted class probabilities.

    proba is an (n_samples, 3) array: the model's probability for each class.
    The predicted class is the one with the highest probability (argmax).
    """
    pred = proba.argmax(axis=1)   # pick the most probable class per student
    # Per-class precision, recall, F1 (arrays of length 3, ordered Dropout/Enrolled/Graduate)
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, labels=[0, 1, 2], zero_division=0)
    onehot = np.eye(3)[y_true]    # true labels as one-hot rows, used for the Brier score
    out = {
        "macro_f1": f1_score(y_true, pred, average="macro"),            # average F1 over classes (main metric)
        "balanced_acc": balanced_accuracy_score(y_true, pred),          # average recall over classes
        "accuracy": accuracy_score(y_true, pred),                       # plain accuracy (misleading with imbalance)
        "log_loss": log_loss(y_true, np.clip(proba, 1e-7, 1), labels=[0, 1, 2]),  # penalizes confident wrong probabilities
        "brier": np.mean(np.sum((proba - onehot) ** 2, axis=1)),        # squared error of probabilities (calibration)
    }
    # Add per-class precision/recall/F1 under names like "Dropout_recall"
    for i, c in enumerate(CLASSES):
        out[f"{c}_precision"], out[f"{c}_recall"], out[f"{c}_f1"] = p[i], r[i], f[i]
    return out


# ----------------------------------------------------------------------------
# 5. TRAIN / TUNE / EVALUATE every (stage, model)
# ----------------------------------------------------------------------------
rows = []          # one dict of results per (stage, model)
best_params = {}   # tuned hyperparameters, saved at the end
fitted = {}        # keep fitted models for the figures below

for stage, cols in STAGES.items():
    # Select only this stage's columns for the train and test students
    X_train = df.iloc[idx_train][cols]
    X_test = df.iloc[idx_test][cols]

    for name, m in MODELS.items():
        # Pipeline = preprocessing + model. Using a pipeline means preprocessing
        # is fit ONLY on the training portion of each CV fold (no leakage from validation rows).
        pipe = Pipeline([("prep", make_preprocessor(cols, m["scale"])), ("clf", clone(m["est"]))])

        if m["params"]:
            # Hyperparameter tuning: try n_iter random combinations, score each with
            # 5-fold CV using macro F1, then refit the best one on the full training set.
            search = RandomizedSearchCV(
                pipe, m["params"], n_iter=m["n_iter"], scoring="f1_macro",
                cv=cv, random_state=SEED, n_jobs=-1, refit=True)
            search.fit(X_train, y_train)
            model = search.best_estimator_     # the best pipeline, already refit on all training data
            # Save best params (convert numpy floats to plain floats so they can be written as JSON)
            best_params[f"{stage} | {name}"] = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                                                for k, v in search.best_params_.items()}
        else:
            # Majority baseline has nothing to tune: just fit it
            model = pipe.fit(X_train, y_train)

        # VALIDATION metrics: out-of-fold predictions on the training set.
        # Every training student is predicted by a model that never saw them.
        # (Slightly optimistic for tuned models, since tuning used the same folds.)
        oof = cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba")
        val = evaluate(y_train, oof)

        # TEST metrics: the model (fit on all training data) predicts the held-out test students.
        test = evaluate(y_test, model.predict_proba(X_test))

        # Store everything for this (stage, model), with "val_" and "test_" prefixes
        row = {"stage": stage, "model": name}
        row.update({f"val_{k}": v for k, v in val.items()})
        row.update({f"test_{k}": v for k, v in test.items()})
        rows.append(row)
        fitted[(stage, name)] = (model, X_test)
        print(f"{stage:22s} {name:13s} val macroF1={val['macro_f1']:.3f} "
              f"test macroF1={test['macro_f1']:.3f} test dropout recall={test['Dropout_recall']:.3f}")

# Save all results to a CSV and the tuned hyperparameters to JSON
res = pd.DataFrame(rows)
res.to_csv(OUT / "results.csv", index=False)
json.dump(best_params, open(OUT / "best_params.json", "w"), indent=2, default=str)

# Compact table for the report (the columns you will most likely paste into a LaTeX table)
cols_report = ["stage", "model", "test_macro_f1", "test_balanced_acc", "test_accuracy",
               "test_Dropout_recall", "test_Dropout_precision", "test_log_loss"]
print("\n", res[cols_report].round(3).to_string(index=False))

# ----------------------------------------------------------------------------
# 6. FIGURES
# ----------------------------------------------------------------------------
# (a) Stage-by-stage macro F1 and dropout recall per model.
# Shows the project's central question: how much does performance improve as
# more information becomes available?
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, metric, title in zip(axes, ["test_macro_f1", "test_Dropout_recall"],
                             ["Test macro F1", "Test dropout recall"]):
    pv = res.pivot(index="stage", columns="model", values=metric)[list(MODELS)]  # rows=stages, columns=models
    pv.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(OUT / "stage_comparison.png", dpi=200)
plt.close()

# (b) Normalized confusion matrices on the test set for tuned GradBoost at each stage.
# normalize="true" -> each row sums to 1, so cells show "of students whose true
# class is X, what fraction was predicted as Y". Good for seeing which classes
# get confused (expect Enrolled to be the hardest).
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

# (c) Per-class report for the best model at each stage.
# "Best" is chosen by VALIDATION macro F1 (not test), so the test set isn't
# used to pick models. The Majority baseline is excluded from the choice.
with open(OUT / "classification_reports.txt", "w") as f:
    for stage in STAGES:
        sub = res[(res.stage == stage) & (res.model != "Majority")]
        best = sub.sort_values("val_macro_f1", ascending=False).iloc[0]["model"]
        model, X_te = fitted[(stage, best)]
        f.write(f"=== {stage} | best by validation macro F1: {best} ===\n")
        f.write(classification_report(y_test, model.predict(X_te), target_names=CLASSES, digits=3))
        f.write("\n")

# (d) Permutation importance for the Stage 3 best model.
# Idea: shuffle one feature's values on the test set and see how much macro F1
# drops. A big drop means the model relies heavily on that feature.
# n_repeats=10 shuffles each feature 10 times and averages for stability.
sub = res[(res.stage == "Stage 3 (+sem 2)") & (res.model != "Majority")]
best3 = sub.sort_values("val_macro_f1", ascending=False).iloc[0]["model"]
model, X_te = fitted[("Stage 3 (+sem 2)", best3)]
pi = permutation_importance(model, X_te, y_test, scoring="f1_macro",
                            n_repeats=10, random_state=SEED, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values().tail(15)  # top 15 features
plt.figure(figsize=(6, 4.5))
imp.plot(kind="barh")
plt.xlabel("Drop in test macro F1 when shuffled")
plt.title(f"Permutation importance, Stage 3 ({best3})")
plt.tight_layout()
plt.savefig(OUT / "permutation_importance_stage3.png", dpi=200)
plt.close()

print("\nDone. See ./outputs/")