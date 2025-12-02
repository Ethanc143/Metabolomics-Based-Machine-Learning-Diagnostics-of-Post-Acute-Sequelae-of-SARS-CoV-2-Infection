import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, RocCurveDisplay, accuracy_score
import matplotlib.pyplot as plt
import sys
from skrebate import ReliefF

#model selection
model_type = "Gradient Boosting"  # Options: "MLP", "Random Forest", "Logistic Regression", "SVM", "Gradient Boosting"
topk = 250

#data processing
pasc_train = pd.read_csv('../pasctraining.csv')
pasc_test = pd.read_csv('../pasctesting.csv')
le = LabelEncoder()
def EncodeSplit(df, first=False):
    x = df.drop(columns=['Class'])
    if first:
        y = le.fit_transform(df['Class'])
    else:
        y = le.transform(df['Class'])
    return x, y

X, Y = EncodeSplit(pasc_train, True)
X_test, Y_test = EncodeSplit(pasc_test)

regenerate = False
if regenerate:
    print("Starting reliefF")
    selector = ReliefF(n_neighbors=100)
    selector.fit(X.values, Y)
    idx_sorted = np.argsort(selector.feature_importances_)[::-1]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("ReliefF Scores")
    plt.plot(range(len(selector.feature_importances_)), selector.feature_importances_[idx_sorted],marker='o')
    plt.show()
    fig.savefig(f"reliefF_scores.pdf", bbox_inches="tight")
    np.savetxt('relieff_selected_features.txt', idx_sorted, fmt='%d')
    sys.exit()
else:
    idx_selected = np.loadtxt('relieff_selected_features.txt', dtype=int)[:topk]

X = X.iloc[:, idx_selected]
X_test = X_test.iloc[:, idx_selected]

#pipeline setup
if model_type == "MLP":
    from sklearn.neural_network import MLPClassifier
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            max_iter=2500,
            solver="adam",
            activation="relu",
            random_state=42
        )),
    ])

    param_grid = {
        "mlp__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32)],
        "mlp__alpha": [0.0001, 0.001, 0.01],
        "mlp__learning_rate_init": [0.0001, 0.001, 0.01]
    }
elif model_type == "Random Forest":
    from sklearn.ensemble import RandomForestClassifier
    pipe = Pipeline([
        ("rf", RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            class_weight=None,
            random_state=42
        )),
    ])

    param_grid = {
        "rf__n_estimators": [100, 300, 500],
        "rf__max_depth": [None, 5, 10, 20],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
        "rf__max_features": ['sqrt', 'log2']
    }
elif model_type == "Logistic Regression":
    from sklearn.linear_model import LogisticRegression
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=5000,
            solver="liblinear",
            random_state=42
        )),
    ])

    param_grid = {
        "logreg__C": [0.01, 0.1, 1.0, 10.0],
        "logreg__penalty": ["l1", "l2"]
    }
elif model_type == "SVM":
    from sklearn.svm import SVC
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        )),
    ])

    param_grid = {
        "svm__C": [0.1, 1.0, 10.0],
        "svm__gamma": ["scale", 0.01, 0.001]
    }
elif model_type == "Gradient Boosting":
    from sklearn.ensemble import GradientBoostingClassifier
    pipe = Pipeline([
        ("gb", GradientBoostingClassifier(
            random_state=42
        )),
    ])

    param_grid = {
        "gb__n_estimators": [100, 300],
        "gb__learning_rate": [0.01, 0.1],
        "gb__max_depth": [2, 3]
    }

#inner CV and grid search
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(estimator=pipe,param_grid=param_grid, scoring="roc_auc", cv=inner_cv, n_jobs=-1, refit=True)
grid.fit(X, Y)
best_model = grid.best_estimator_

#outer CV for performance estimation
outer_cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42,
)

scores = cross_val_score(
    best_model,
    X,
    Y,
    scoring="roc_auc",
    cv=outer_cv,
    n_jobs=-1,
)

#report statistics
mean_auc = scores.mean()
ci_lower = np.percentile(scores, 2.5)
ci_upper = np.percentile(scores, 97.5)

print(f"Mean ROC-AUC: {mean_auc:.3f}")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

#training and testing predictions
y_train_proba = best_model.predict_proba(X)[:, 1]  
y_test_proba = best_model.predict_proba(X_test)[:, 1]  
test_auc = roc_auc_score(Y_test, y_test_proba)
print(f"Test ROC-AUC: {test_auc:.3f}")

def bootstrapped_accuracy_ci(model, X_test, Y_test, B=500):
    n = len(Y_test)
    accs = []

    for _ in range(B):
        idx = np.random.randint(0, n, n)
        Xb = X_test.iloc[idx]
        Yb = Y_test[idx]
        pred = model.predict(Xb)
        accs.append((pred == Yb).mean())

    accs = np.array(accs)
    return np.mean(accs), np.percentile(accs, [2.5, 97.5])

pasc_idx = (Y_test == 0)
X_test_pasc = X_test[pasc_idx]
Y_test_pasc = Y_test[pasc_idx]
mean, ci = bootstrapped_accuracy_ci(best_model, X_test_pasc, Y_test_pasc)
print(f"PASC testing results: Mean Accuracy = {mean:.3f}, 95% CI = [{ci[0]:.3f}, {ci[1]:.3f}]")

other_diseases = ["descriptors_FM", "descriptors_IBS", "descriptors_lyme_disease", "descriptors_ME_SFC", "descriptors_POTS"]
for disease in other_diseases:
    disease_set = pd.read_csv(f'{disease}.csv')
    X_test_disease, Y_test_disease = EncodeSplit(disease_set)
    X_test_disease = X_test_disease.iloc[:, idx_selected]
    mean, ci = bootstrapped_accuracy_ci(best_model, X_test_disease, Y_test_disease)
    print(f"{disease} testing results: Mean Accuracy = {mean:.3f}, 95% CI = [{ci[0]:.3f}, {ci[1]:.3f}]")

#draw and save AUC curves
fig, ax = plt.subplots(figsize=(6, 6))
RocCurveDisplay.from_predictions(
    Y,
    y_train_proba,
    ax=ax,
    name="Train ROC"
)
RocCurveDisplay.from_predictions(
    Y_test,
    y_test_proba,
    ax=ax,
    name="Test ROC"
)
ax.plot([0, 1], [0, 1], "k--", label="Chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"{model_type} ROC Curves - Train vs Test")
ax.legend(loc="lower right")
plt.show()
fig.savefig(f"{model_type}_roc_train_test.pdf", bbox_inches="tight")
    

