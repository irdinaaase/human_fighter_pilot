import tensorflow as tf
print(tf.__version__)  # Should print 2.18.0
import keras
print(keras.__version__)  # Should print 3.8.0

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
from xgboost.sklearn import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from collections import Counter

# Print library versions
print(f"xgboost version: {xgboost.__version__}")
print(f"sklearn version: {sklearn.__version__}")

# Load the dataset
file_path = 'STTHK3013_pilot_performance_simulation_data.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')
print(f"\nData shape: {data.shape}")

# Handle missing values in the target column
data.dropna(subset=['final_performance'], inplace=True)

# Plot distribution of each variable
columns = data.columns
plt.figure(figsize=(16, 20))

print(f"\ndata shape: {data.shape}") 

# Adjust subplot layout dynamically for better spacing
plt.figure(figsize=(15, len(columns)))  # Adjust figure size based on the number of columns

for i, col in enumerate(columns, 1):
    plt.subplot((len(columns) + 2) // 3, 3, i)  # Arrange plots in a grid
    if data[col].nunique() < 20:  # Discrete/categorical
        sns.countplot(x=col, data=data, palette="viridis")  # Explicitly set x and data
        plt.title(f'Distribution of {col} (Categorical)', fontsize=12)
    else:  # Continuous
        sns.histplot(data[col], kde=True, bins=30, color='blue')
        plt.title(f'Distribution of {col} (Continuous)', fontsize=12)
    
    plt.xlabel(col, fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

plt.tight_layout()
plt.show()
print(f"\ndata shape: {data.shape}") 

# Columns that require mean imputation
continuous_columns = [
    'heart_rate', 'sleep_quality', 'mission_complexity', 'experience_level',
    'environmental_stressors', 'cognitive_level', 'fatigue_level', 'stress_level', 'time_reaction'
]
for col in continuous_columns:
    knn_imputer = KNNImputer(n_neighbors=5)  # You can change the number of neighbors as needed
    data[col] = knn_imputer.fit_transform(data[[col]]).flatten()  # Use fit_transform to get imputed data and flatten to assign to column

# Final check for any remaining NaN values
if data.isnull().values.any():
    print("\nWarning: NaN values detected after initial processing. Imputing again...")
    data.fillna(data.mean(), inplace=True)

# Map experience level to categorical labels
def map_experience_level(value):
    if 0 <= value <= 3:
        return 0
    elif 4 <= value <= 7:
        return 1
    else: 
        return 2
data['experience_level'].apply(map_experience_level)

# Map sleep quality to categorical labels
def map_sleep_quality(value):
    if 0 <= value <= 3:
        return 0
    elif 4 <= value <= 7:
        return 1
    else:
        return 2
data['sleep_quality'].apply(map_sleep_quality)

# Analyze and plot the distribution of the target variable 'final_performance'
target_distribution = data['final_performance'].value_counts().sort_index()

# Plot the distribution of 'final_performance'
plt.figure(figsize=(10, 6))
sns.barplot(x=target_distribution.index, y=target_distribution.values, palette="viridis")
plt.title('Distribution of Final Performance', fontsize=16)
plt.xlabel('Final Performance', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Normalize features
scaler = StandardScaler()

# Identify numeric columns to scale
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# Print class distribution before SMOTE
print("\nOriginal Class Distribution:", Counter(y))

# Apply SMOTE with sampling strategy
smote = SMOTE(sampling_strategy='auto', random_state=42)

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

    #------------------------ Logistic Regression ------------------------
    log_reg_model = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), log_reg_param_grid, cv=3, scoring='accuracy')
    log_reg_model.fit(X_train_smote, y_train_smote)
    y_pred_log_reg = log_reg_model.best_estimator_.predict(X_test)
    log_reg_accuracies.append(accuracy_score(y_test, y_pred_log_reg))

    #------------------------ XG Boost ------------------------
    xgb_model = GridSearchCV(XGBClassifier(random_state=42, eval_metric='mlogloss'), xgb_param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    xgb_model.fit(X_train_smote, y_train_smote)
    y_pred_xgb = xgb_model.best_estimator_.predict(X_test)
    xgb_accuracies.append(accuracy_score(y_test, y_pred_xgb))

    #------------------------ Deep Neural Network (DNN) ------------------------
    num_classes = len(np.unique(y))
    y_train_encoded = to_categorical(y_train_smote, num_classes=num_classes)
    y_test_encoded = to_categorical(y_test, num_classes=num_classes)

    dnn_model = Sequential([ 
        Dense(128, activation='relu', input_dim=X_train_smote.shape[1]), 
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    dnn_model.fit(X_train_smote, y_train_encoded, validation_split=0.2, epochs=20, batch_size=32, verbose=0)
    _, dnn_accuracy = dnn_model.evaluate(X_test, y_test_encoded, verbose=0)
    dnn_accuracies.append(dnn_accuracy)

# Calculate and print the average accuracy for each model
print("\nModel Performance Summary:")
print("Logistic Regression Average Accuracy:", np.mean(log_reg_accuracies))
print("XGBoost Average Accuracy:", np.mean(xgb_accuracies))
print("DNN Average Accuracy:", np.mean(dnn_accuracies))
