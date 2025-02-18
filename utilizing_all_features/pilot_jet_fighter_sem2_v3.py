# -------------------- LIBRARIES --------------------
import tensorflow as tf
import joblib
import keras
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from imblearn.over_sampling import SMOTE
from collections import Counter

# -------------------- LOAD DATA --------------------
file_path = r"C:\Users\Irdina Balqis\Documents\GitHub\human_fighter_pilot\dataset\STTHK3013_pilot_performance_simulation_data.xlsx"
data = pd.read_excel(file_path, sheet_name='Sheet1')

# Remove rows with missing target values
data.dropna(subset=['final_performance'], inplace=True)

# -------------------- CLASS CONSOLIDATION --------------------
def categorize_performance(value):
    if value <= 1:
        return 0  # Low
    elif 2 <= value <= 3:
        return 1  # Medium
    elif 4 <= value <= 5:
        return 2  # High

data['performance_category'] = data['final_performance'].apply(categorize_performance)

# -------------------- DEFINE FEATURES & TARGET --------------------
X = data.drop(columns=['performance_category', 'final_performance'])
y = data['performance_category']

# -------------------- TRAIN-TEST SPLIT FIRST (To Avoid Data Leakage) --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# -------------------- MISSING VALUE IMPUTATION --------------------
continuous_columns = [
    'heart_rate', 'sleep_quality', 'mission_complexity', 'experience_level',
    'environmental_stressors', 'cognitive_level', 'fatigue_level', 'stress_level', 'time_reaction'
]
knn_imputer = KNNImputer(n_neighbors=5)
X_train[continuous_columns] = knn_imputer.fit_transform(X_train[continuous_columns])
X_test[continuous_columns] = knn_imputer.transform(X_test[continuous_columns])  # Avoid leakage

# -------------------- FEATURE ENGINEERING (After Splitting) --------------------
X_train["fatigue_stress_index"] = X_train["fatigue_level"] * X_train["stress_level"]
X_train["cognitive_experience_score"] = X_train["cognitive_level"] * X_train["experience_level"]
X_train["adaptability_score"] = X_train["experience_level"] / (X_train["environmental_stressors"] + X_train["mission_complexity"] + 1)
X_train["normalized_reaction_time"] = np.log1p(X_train["time_reaction"] / X_train["time_reaction"].max())

X_test["fatigue_stress_index"] = X_test["fatigue_level"] * X_test["stress_level"]
X_test["cognitive_experience_score"] = X_test["cognitive_level"] * X_test["experience_level"]
X_test["adaptability_score"] = X_test["experience_level"] / (X_test["environmental_stressors"] + X_test["mission_complexity"] + 1)
X_test["normalized_reaction_time"] = np.log1p(X_test["time_reaction"] / X_test["time_reaction"].max())

# # Drop redundant features
# drop_cols = ['fatigue_level', 'stress_level', 'cognitive_level', 'experience_level', 'environmental_stressors', 'time_reaction', 'mission_complexity']
# X_train.drop(columns=drop_cols, inplace=True)
# X_test.drop(columns=drop_cols, inplace=True)

# -------------------- NORMALIZATION --------------------
scaler = StandardScaler()
X_train[X_train.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(X_train)
X_test[X_test.select_dtypes(include=['float64', 'int64']).columns] = scaler.transform(X_test)

# Save scaler for GUI usage
joblib.dump(scaler, "scaler.pkl")
print("✅  saved as scaler.pkl")

# -------------------- APPLY SMOTE ONLY ON TRAIN DATA --------------------
smote = SMOTE(sampling_strategy={2: 358, 1: 352, 0: 327}, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nClass Distribution After SMOTE:", Counter(y_train_smote))

# -------------------- K-FOLD CROSS-VALIDATION --------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

log_reg_accuracies = []
xgb_accuracies = []
dnn_accuracies = []

log_reg_param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga']}
xgb_param_grid = {'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7], 'n_estimators': [50, 100, 200], 'subsample': [0.8, 0.9, 1.0]}

# -------------------- MODEL TRAINING & TESTING --------------------
for train_index, test_index in kf.split(X_train_smote, y_train_smote):
    X_k_train, X_k_test = X_train_smote.iloc[train_index], X_train_smote.iloc[test_index]
    y_k_train, y_k_test = y_train_smote.iloc[train_index], y_train_smote.iloc[test_index]

    # -------------------- LOGISTIC REGRESSION --------------------
    log_reg_model = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), log_reg_param_grid, cv=3, scoring='accuracy')
    log_reg_model.fit(X_k_train, y_k_train)
    y_pred_log_reg = log_reg_model.best_estimator_.predict(X_k_test)
    log_reg_accuracies.append(accuracy_score(y_k_test, y_pred_log_reg))

    # -------------------- XGBOOST --------------------
    xgb_model = GridSearchCV(XGBClassifier(random_state=42, eval_metric='mlogloss'), xgb_param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    xgb_model.fit(X_k_train, y_k_train)
    y_pred_xgb = xgb_model.best_estimator_.predict(X_k_test)
    xgb_accuracies.append(accuracy_score(y_k_test, y_pred_xgb))

    # -------------------- DEEP NEURAL NETWORK (DNN) --------------------
    num_classes = len(np.unique(y_train_smote))

    y_k_train_encoded = to_categorical(y_k_train, num_classes=num_classes)
    y_k_test_encoded = to_categorical(y_k_test, num_classes=num_classes)

    dnn_model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(0.01), input_dim=X_k_train.shape[1]),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    dnn_model.fit(X_k_train, y_k_train_encoded, validation_split=0.2, epochs=20, batch_size=32, verbose=0)

    _, dnn_accuracy = dnn_model.evaluate(X_k_test, y_k_test_encoded, verbose=0)
    dnn_accuracies.append(dnn_accuracy)

# -------------------- PRINT FINAL RESULTS --------------------
print("\nModel Performance Summary:")
print(f"Logistic Regression Average Accuracy: {np.mean(log_reg_accuracies):.4f}")

print(f"XGBoost Average Accuracy: {np.mean(xgb_accuracies):.4f}")

# Get the best model from GridSearchCV
best_xgb_model = xgb_model.best_estimator_

# Save the model
joblib.dump(best_xgb_model, "xgb_best_model.pkl")
print("✅ XGBoost model saved as xgb_best_model.pkl")

print(f"DNN Average Accuracy: {np.mean(dnn_accuracies):.4f}")
