import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import xgboost
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
from imblearn.over_sampling import SMOTE
from collections import Counter

# Print library versions
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
print(f"xgboost version: {xgboost.__version__}")
print(f"sklearn version: {sklearn.__version__}")

# ---------------------------- Load Data ----------------------------
file_path = r"C:\Users\Irdina Balqis\Documents\GitHub\human_fighter_pilot\dataset\STTHK3013_pilot_performance_simulation_data.xlsx"
data = pd.read_excel(file_path, sheet_name='Sheet1')
print(f"\nData shape: {data.shape}")

# Handle missing values in the target column
data.dropna(subset=['final_performance'], inplace=True)

# ---------------------------- Should Update: Class Consolidation ----------------------------
def categorize_performance(value):
    """Convert numerical performance scores into categories."""
    if value <= 1:
        return 0  # Low
    elif 2 <= value <= 3:
        return 1  # Medium
    elif 4 <= value <= 5:
        return 2  # High

# Apply class consolidation
data['performance_category'] = data['final_performance'].apply(categorize_performance)

# Check updated distribution
print("Updated Class Distribution:", data['performance_category'].value_counts())

# ---------------------------- Data Preprocessing ----------------------------

# Columns that require KNN imputation
continuous_columns = [
    'heart_rate', 'sleep_quality', 'mission_complexity', 'experience_level',
    'environmental_stressors', 'cognitive_level', 'fatigue_level', 'stress_level', 'time_reaction'
]
knn_imputer = KNNImputer(n_neighbors=5)
data[continuous_columns] = knn_imputer.fit_transform(data[continuous_columns])

# Final check for remaining NaN values
if data.isnull().values.any():
    print("\nWarning: NaN values detected after initial processing. Imputing again...")
    data.fillna(data.mean(), inplace=True)

# ---------------------------- Feature Engineering ----------------------------
data["fatigue_stress_index"] = data["fatigue_level"] * data["stress_level"]
data["cognitive_experience_score"] = data["cognitive_level"] * data["experience_level"]
data["adaptability_score"] = data["experience_level"] / (data["environmental_stressors"] + data["mission_complexity"] + 1)
data["normalized_reaction_time"] = np.log1p(data["time_reaction"] / data["time_reaction"].max())


# Drop redundant features after engineering
data.drop(columns=['fatigue_level', 'stress_level', 'cognitive_level', 'experience_level'], inplace=True)

# Define features
X = data.drop(columns=['performance_category'])
y = data['performance_category']

# Normalize numerical features
scaler = StandardScaler()
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# Print class distribution before SMOTE
print("\nOriginal Class Distribution:", Counter(y))

# Apply SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy={0: 358, 1: 352, 2: 327}, random_state=42)

# K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store accuracies
log_reg_accuracies = []
xgb_accuracies = []
dnn_accuracies = []

# Parameter grid for Logistic Regression
log_reg_param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Parameter grid for XGBoost
xgb_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 0.9, 1.0]
}

# Loop through each fold
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Apply SMOTE only to the training set
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"Class Distribution After SMOTE (Train): {Counter(y_train_smote)}")

    # ------------------------ Logistic Regression ------------------------
    log_reg_model = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), log_reg_param_grid, cv=3, scoring='accuracy')
    log_reg_model.fit(X_train_smote, y_train_smote)
    y_pred_log_reg = log_reg_model.best_estimator_.predict(X_test)
    log_reg_accuracies.append(accuracy_score(y_test, y_pred_log_reg))

    # ------------------------ XGBoost ------------------------
    xgb_model = GridSearchCV(XGBClassifier(random_state=42, eval_metric='mlogloss'), xgb_param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    xgb_model.fit(X_train_smote, y_train_smote)
    y_pred_xgb = xgb_model.best_estimator_.predict(X_test)
    xgb_accuracies.append(accuracy_score(y_test, y_pred_xgb))

    # ------------------------ Deep Neural Network (DNN) ------------------------
    num_classes = len(np.unique(y))

    # Convert labels to categorical format
    y_train_encoded = to_categorical(y_train_smote, num_classes=num_classes)
    y_test_encoded = to_categorical(y_test, num_classes=num_classes)

    # Define DNN model
    dnn_model = Sequential([
        Dense(128, activation='relu', input_dim=X_train_smote.shape[1]),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the DNN model
    dnn_model.fit(X_train_smote, y_train_encoded, validation_split=0.2, epochs=20, batch_size=32, verbose=0)

    # Evaluate DNN model
    _, dnn_accuracy = dnn_model.evaluate(X_test, y_test_encoded, verbose=0)
    dnn_accuracies.append(dnn_accuracy)

# Print final model performance
print("\nModel Performance Summary:")
print(f"Logistic Regression Average Accuracy: {np.mean(log_reg_accuracies):.4f}")
print(f"XGBoost Average Accuracy: {np.mean(xgb_accuracies):.4f}")
print(f"DNN Average Accuracy: {np.mean(dnn_accuracies):.4f}")
