import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# importando os dados treino eteste
brain_mri_train = pd.read_csv("brain_mri_train.csv")
brain_mri_test = pd.read_csv("brain_mri_test.csv")

X_train = brain_mri_train.drop('label', axis=1)
y_train = brain_mri_train['label']

X_test = brain_mri_test.drop('label', axis=1)
y_test = brain_mri_test['label']

print(X_train.head())

# Validação Cruzada Estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# modelo SVM

pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(class_weight='balanced'))
])

param_grid_svm = {
    'model__C': [0.1, 1, 10],
    'model__kernel': ['linear', 'rbf'],
    'model__gamma': ['scale', 'auto']
}

grid_svm = GridSearchCV(
    pipeline_svm,
    param_grid_svm,
    cv=cv,
    scoring= 'f1',
    n_jobs=-1
)

grid_svm.fit(X_train, y_train)

print("SVM\n")
print(grid_svm.best_params_)
print(grid_svm.best_score_)
pred_svm = grid_svm.best_estimator_.predict(X_test)
print(classification_report(y_test, pred_svm, target_names=['Sem tumor', 'Tumor']))

ConfusionMatrixDisplay.from_estimator(
    grid_svm.best_estimator_,
    X_test,
    y_test,
    display_labels=['Sem tumor', 'Tumor'],
    cmap='Blues'
)

plt.title("Matriz de Confusão — SVM")
plt.show()

# modelo KNN

pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier())
])

param_grid_knn = {
    'model__n_neighbors': [3, 5, 7, 9, 10],
    'model__weights': ['uniform', 'distance'],
    'model__metric': ['euclidean', 'manhattan']
}

grid_knn = GridSearchCV(
    pipeline_knn,
    param_grid_knn,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)

grid_knn.fit(X_train, y_train)

print("KNN\n")
print(grid_knn.best_params_)
print(grid_knn.best_score_)
pred_knn = grid_knn.best_estimator_.predict(X_test)
print(classification_report(y_test, pred_knn, target_names=['Sem tumor', 'Tumor']))

ConfusionMatrixDisplay.from_estimator(
    grid_knn.best_estimator_,
    X_test,
    y_test,
    display_labels=['Sem tumor', 'Tumor'],
    cmap='Blues'
)

plt.title("Matriz de Confução - KNN")
plt.show()

# modelo Regrssão logistica

pipeline_lr = Pipeline ([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000,
        class_weight='balanced'))
])

param_grid_lr = {
    'model__C': [0.01, 0.1, 1, 10],
}

grid_lr = GridSearchCV(
    pipeline_lr,
    param_grid_lr,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)

grid_lr.fit(X_train, y_train)

print('LogisticRegression\n')
print((grid_lr.best_params_))
print(grid_lr.best_score_)
pred_lr = grid_lr.best_estimator_.predict(X_test)
print(classification_report(y_test, pred_lr, target_names=['Sem tumor', 'Tumor']))

ConfusionMatrixDisplay.from_estimator(
    grid_lr.best_estimator_,
    X_test,
    y_test,
    display_labels=['Sem tumor', 'Tumor'],
    cmap='Blues'
)

plt.title("Matriz de Confução - LogisticRegression")
plt.show()

# modelo Random Forest

pipeline_rf = Pipeline([
    ('model', RandomForestClassifier(
        random_state=42,
        class_weight='balanced'
    ))
])

param_grid_rf = {
    'model__n_estimators': [200],
    'model__max_depth': [5, 10],
    'model__min_samples_split': [2, 4],
    'model__min_samples_leaf': [5, 10]
}

grid_rf = GridSearchCV(
    pipeline_rf,
    param_grid_rf,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)

grid_rf.fit(X_train, y_train)

print("Random Forest\n")
print(grid_rf.best_params_)
print(grid_rf.best_score_)
pred_rf = grid_rf.best_estimator_.predict(X_test)
print(classification_report(y_test, pred_rf, target_names=['Sem tumor', 'Tumor']))

ConfusionMatrixDisplay.from_estimator(
    grid_rf.best_estimator_,
    X_test,
    y_test,
    display_labels=['Sem tumor', 'Tumor'],
    cmap='Blues'
)

plt.title("Matriz de Confução - Random Forest")
plt.show()
