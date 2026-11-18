# Comparing 5 Supervised Learning Algorithms on Titanic Survival

A fair, side-by-side comparison of five classifiers trained on the **same dataset, same train/test split and same preprocessing**.

![model comparison](model_comparison.png)

## Dataset

[Titanic passenger data](https://github.com/datasciencedojo/datasets) (891 passengers, 38.4% survived). Task: predict `Survived` (binary classification).

It is a good test bed because it isn't trivially separable (unlike Iris) and it has the messy features the task asks us to handle:

| Feature | Type | Issue handled |
|---|---|---|
| `Age` | numeric | 177 missing values → median imputation |
| `Embarked` | categorical | 2 missing values → most-frequent imputation, one-hot encoded |
| `Sex` | categorical | one-hot encoded |
| `Pclass`, `SibSp`, `Parch`, `Fare` | numeric | standardized |

Dropped: `PassengerId`, `Name`, `Ticket` (identifiers) and `Cabin` (77% missing).

## Method

- **Split:** 80/20 stratified (`random_state=42`) → 712 train / 179 test. All five models use this exact split.
- **Preprocessing** (one `ColumnTransformer` inside a `Pipeline`, fit on the training set only, so there is no test leakage): median impute + `StandardScaler` for numerics, most-frequent impute + `OneHotEncoder` for categoricals.
- **Scaling for every model:** KNN, SVM and logistic regression need it; the tree and Naive Bayes don't, but applying identical preprocessing to all five keeps the comparison fair (it doesn't hurt them).
- **Models:** scikit-learn defaults, no hyperparameter tuning (`LogisticRegression`, unpruned `DecisionTreeClassifier`, `KNeighborsClassifier(k=5)`, `SVC(kernel="rbf")`, `GaussianNB`).
- **Metrics:** accuracy, precision, recall, F1 and ROC-AUC on the test set. Models are ranked by **test F1** (the positive class, survived, is the minority, so F1 is more informative than accuracy alone). As a robustness check I also report 5-fold cross-validated accuracy on the training set and training accuracy (to expose overfitting).

## Results

| Rank | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train Acc | CV Acc (5-fold) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | K-Nearest Neighbors | 0.8156 | 0.8000 | 0.6957 | **0.7442** | 0.8495 | 0.8511 | 0.7977 ± 0.0164 |
| 2 | Decision Tree | 0.8045 | 0.7656 | 0.7101 | 0.7368 | 0.7808 | 0.9831 | 0.7655 ± 0.0163 |
| 3 | SVM (RBF) | 0.8156 | 0.8214 | 0.6667 | 0.7360 | 0.8401 | 0.8427 | 0.8216 ± 0.0247 |
| 4 | Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8437 | 0.8076 | 0.7963 ± 0.0278 |
| 5 | Naive Bayes (Gaussian) | 0.7877 | 0.7385 | 0.6957 | 0.7164 | 0.8191 | 0.7907 | 0.7936 ± 0.0301 |

Confusion matrices on the test set, as `[[TN, FP], [FN, TP]]`:
Logistic Regression `[[98, 12], [23, 46]]` · Decision Tree `[[95, 15], [20, 49]]` · KNN `[[98, 12], [21, 48]]` · SVM `[[100, 10], [23, 46]]` · Naive Bayes `[[93, 17], [21, 48]]`.

Raw numbers: `results.csv`, `confusion_matrices.json`, `run_output.txt`.

### An honest caveat first

The five models are **close**. With only 179 test passengers, the standard error on an accuracy of ~0.80 is about ±3 percentage points, and the gap between best and worst accuracy is 2.8 points (5 passengers). So the ranking above shows tendencies, not a statistically solid ordering, and a different random split could reshuffle the middle of the table. The cross-validation column is a useful second opinion, and it doesn't fully agree with the test ranking (SVM is best on CV, the decision tree worst). The explanations below are hypotheses that fit the data, not results I proved with ablations.

## Analysis

### Why K-Nearest Neighbors came out on top (and why SVM is a close co-winner)

KNN has the best test F1 (0.744) and ROC-AUC (0.850), and ties SVM for best accuracy (0.8156). The RBF SVM has the best cross-validated accuracy (0.822) and the best precision. Both are *local / non-linear* methods, and survival on the Titanic is driven by **interactions** rather than independent effects: women and children in 1st/2nd class survived at very high rates, men in 3rd class at very low rates, and the effect of age depends on sex. A model that groups "passengers like this one" captures these combinations without being told about them.

The data also suits distance-based methods: after one-hot encoding there are only 9 features, so distances remain meaningful (no curse of dimensionality), and the standardization step puts `Age`, `Fare` and `Pclass` on comparable scales so no single feature dominates the distance. Averaging over 5 neighbors and the smooth RBF kernel both give some noise tolerance, which matters because the data is noisy (survival involved a lot of luck that no feature records).

### Why Naive Bayes struggled most on the test set

Gaussian Naive Bayes has the lowest accuracy (0.788), F1 (0.716) and precision (0.739), mostly from extra false positives (17 versus 10–15 for the others). Its core assumption, that features are independent *given the class*, is badly violated here:

- `Fare` and `Pclass` are strongly correlated (first class pays more), and `Embarked` correlates with class, so the same evidence gets counted more than once.
- The one-hot columns for a single variable (`Sex_female`/`Sex_male`, the three `Embarked_*` columns) are perfectly dependent on each other.
- It models every feature as a Gaussian per class, but binary dummies and the heavy-tailed, right-skewed `Fare` aren't Gaussian.

Those mismatches distort its probability estimates. It also cannot represent the Sex × Class × Age interactions that matter most. In its favor, its cross-validated accuracy (0.794) is almost the same as logistic regression and KNN, which is another sign the gaps are small.

### The other two models

- **Decision Tree:** the clearest case of overfitting. Training accuracy is 98.3% against 80.5% on the test set and 76.6% in cross-validation. Left unpruned, it memorizes noise in individual passengers. It also gives the worst ROC-AUC (0.781) because its probabilities come from tiny, nearly pure leaves, which rank passengers poorly even when the hard 0/1 predictions look fine. Its test accuracy (0.8045) probably benefits from a lucky split; I'd expect a depth-limited or pruned tree to do better, though I did not tune it here.
- **Logistic Regression:** a linear boundary in the (scaled, encoded) features can't express interactions without hand-made interaction terms, yet it is competitive (accuracy 0.8045, AUC 0.844) because `Sex` alone explains most of the signal. It is also the simplest and most interpretable model, with a small train/test gap, so it is the safest choice if explainability matters.

## Takeaways

1. On low-dimensional tabular data with strong interactions, local and kernel methods (KNN, RBF-SVM) have an edge over linear or independence-assuming models, but the margin here is small.
2. Training accuracy is a poor guide: the tree "won" on the training set (98%) and lost in cross-validation.
3. Always check variance. On 179 test samples, ±3 points of noise is as large as the whole spread between the five models.

## Run it

```bash
pip install -r requirements.txt
python compare_models.py
```

This regenerates `model_comparison.png`, `results.csv` and `confusion_matrices.json`. Results are reproducible (`random_state=42` everywhere).

## Project layout

| File | Purpose |
|---|---|
| `compare_models.py` | Preprocessing pipeline, trains and evaluates all 5 models, builds the chart |
| `data/titanic.csv` | Dataset |
| `results.csv`, `confusion_matrices.json`, `run_output.txt` | Output of the run reported above |
| `model_comparison.png` | Comparison chart |
