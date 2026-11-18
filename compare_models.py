"""Compare 5 supervised classifiers on the Titanic survival dataset.

Every model sees the SAME train/test split and the SAME preprocessing:
  * numeric  -> median imputation (fit on train) -> standard scaling (fit on train)
  * categorical -> most-frequent imputation (fit on train) -> one-hot encoding
Scaling is applied to all models for a fair comparison; it matters for
Logistic Regression, KNN and SVM, and is harmless for the tree and Naive Bayes.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
SEED = 42

NUMERIC = ["Age", "SibSp", "Parch", "Fare", "Pclass"]
CATEGORICAL = ["Sex", "Embarked"]
TARGET = "Survived"


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    # sparse_threshold=0 -> dense output (GaussianNB needs dense input)
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)],
                             sparse_threshold=0)


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": SVC(kernel="rbf", random_state=SEED),
    "Naive Bayes (Gaussian)": GaussianNB(),
}


def main():
    df = pd.read_csv(ROOT / "data" / "titanic.csv")
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    print(f"rows={len(df)} | missing: Age={df.Age.isna().sum()}, Embarked={df.Embarked.isna().sum()} "
          f"| survival rate={y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"train={len(X_train)} test={len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, matrices = [], {}
    for name, clf in MODELS.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", clf)])
        pipe.fit(X_train, y_train)  # imputers/scalers/encoders fit on train only
        pred = pipe.predict(X_test)
        # ROC-AUC needs a continuous score: probabilities if the model has them, else the SVM margin
        score = (pipe.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba")
                 else pipe.decision_function(X_test))
        cv_acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, score),
            "Train Acc": accuracy_score(y_train, pipe.predict(X_train)),
            "CV Acc (5-fold, train)": cv_acc.mean(),
            "CV Std": cv_acc.std(),
        })
        matrices[name] = confusion_matrix(y_test, pred).tolist()

    res = pd.DataFrame(rows).sort_values(["F1", "Accuracy"], ascending=False).reset_index(drop=True)
    res.insert(0, "Rank", res.index + 1)
    res.to_csv(ROOT / "results.csv", index=False)
    (ROOT / "confusion_matrices.json").write_text(json.dumps(matrices, indent=2))
    print("\n" + res.round(4).to_string(index=False))
    print("\nConfusion matrices [[TN, FP],[FN, TP]]:")
    for k, v in matrices.items():
        print(f"  {k}: {v}")

    # ---- Chart: grouped bars of test metrics, ordered by rank ----
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    ax = res.set_index("Model")[metrics].plot.bar(figsize=(11, 5.5), width=0.8, rot=15)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score (test set)")
    ax.set_title("Titanic survival: 5 classifiers on the same split, ranked by F1")
    ax.legend(loc="upper center", ncol=5, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "model_comparison.png", dpi=150)


if __name__ == "__main__":
    main()
