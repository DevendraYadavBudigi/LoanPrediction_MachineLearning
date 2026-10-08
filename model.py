import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

# 1. Load Dataset
df = pd.read_csv('loan_approval_dataset.csv')

# 2. Data Cleaning
# Strip leading and trailing whitespace from column names and string fields
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

# 3. Encode Target and Categorical Features
# Target encoding: Approved -> 1, Rejected -> 0
df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})

# Feature encoding
df['education'] = df['education'].map({'Not Graduate': 0, 'Graduate': 1})
df['self_employed'] = df['self_employed'].map({'No': 0, 'Yes': 1})

# Separate features (X) and target (y), dropping unique identifier
X = df.drop(columns=['loan_id', 'loan_status'], errors='ignore')
y = df['loan_status']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Initialize and Train XGBoost Classifier
model = XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# 6. Model Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("ROC-AUC Score:", round(roc_auc_score(y_test, y_proba), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Feature Importance Analysis
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:\n", feature_importance.to_string(index=False))