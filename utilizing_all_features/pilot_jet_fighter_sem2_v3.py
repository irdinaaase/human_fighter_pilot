import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import xgboost
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats.mstats import winsorize
from tensorflow.keras.utils import to_categorical
from xgboost.sklearn import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from imblearn.over_sampling import SMOTE

# ---------------------------- Load Data ----------------------------
file_path = r"C:\Users\Irdina Balqis\Documents\GitHub\human_fighter_pilot\dataset\STTHK3013_pilot_performance_simulation_data.xlsx"
data = pd.read_excel(file_path, sheet_name='Sheet1')
print(f"\nData shape: {data.shape}")

# Handle missing values in the target column
data.dropna(subset=['final_performance'], inplace=True)

# ---------------------------- Class Consolidation ----------------------------
def categorize_performance(value):
    if value <= 1:
        return 0  # Low
    elif 2 <= value <= 3:
        return 1  # Medium
    else:
        return 2  # High

data['performance_category'] = data['final_performance'].apply(categorize_performance)

# ---------------------------- Data Preprocessing ----------------------------
continuous_columns = [
    'heart_rate', 'sleep_quality', 'mission_complexity', 'experience_level',
    'environmental_stressors', 'cognitive_level', 'fatigue_level', 'stress_level', 'time_reaction'
]

# Apply KNN Imputer
knn_imputer = KNNImputer(n_neighbors=5)
data[continuous_columns] = knn_imputer.fit_transform(data[continuous_columns])

# Handle Outliers
def remove_outliers_iqr(df, cols):
    for col in cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)
    return df

outlier_cols = ['heart_rate', 'sleep_quality', 'experience_level', 'cognitive_level', 'stress_level', 'time_reaction']
data = remove_outliers_iqr(data, outlier_cols)
data[outlier_cols] = data[outlier_cols].apply(lambda col: winsorize(col, limits=(0.05, 0.05)))

# Feature Engineering
data['high_stress_alert'] = ((data['stress_level'] > data['stress_level'].quantile(0.90)) & 
                           (data['heart_rate'] > data['heart_rate'].quantile(0.90))).astype(int)

data['low_sleep_risk'] = ((data['sleep_quality'] < data['sleep_quality'].quantile(0.10)) & 
                        (data['fatigue_level'] > data['fatigue_level'].quantile(0.75))).astype(int)

data['elite_experience'] = ((data['experience_level'] > data['experience_level'].quantile(0.95)) & 
                          (data['cognitive_level'] > data['cognitive_level'].quantile(0.80))).astype(int)

# Normalization & Standardization
scaler = MinMaxScaler()
std_scaler = StandardScaler()
data['normalized_reaction_time'] = scaler.fit_transform(data[['time_reaction']])
data['std_cognitive_level'] = std_scaler.fit_transform(data[['cognitive_level']])

# Drop unnecessary features
drop_cols = ['heart_rate', 'sleep_quality', 'experience_level', 'cognitive_level', 
             'stress_level', 'time_reaction', 'fatigue_level', 'mission_complexity', 'environmental_stressors']
data.drop(columns=drop_cols, inplace=True)

# ---------------------------- Train-Test Split ----------------------------
X = data.drop(columns=['performance_category'])
y = data['performance_category']

# Split the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ---------------------------- Model Training ----------------------------

# Logistic Regression
log_reg_param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga']}
log_reg_model = GridSearchCV(LogisticRegression(max_iter=5000), log_reg_param_grid, cv=3, scoring='accuracy')
log_reg_model.fit(X_train, y_train)
log_reg_pred = log_reg_model.best_estimator_.predict(X_test)
log_reg_acc = accuracy_score(y_test, log_reg_pred)
print(f"\nLogistic Regression Accuracy: {log_reg_acc:.4f}")

# XGBoost
xgb_param_grid = {'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7], 'n_estimators': [50, 100, 200]}
xgb_model = GridSearchCV(XGBClassifier(eval_metric='mlogloss'), xgb_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.best_estimator_.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"\nXGBoost Accuracy: {xgb_acc:.4f}")

# DNN Model
num_classes = len(np.unique(y))
y_train_encoded = to_categorical(y_train, num_classes)
y_test_encoded = to_categorical(y_test, num_classes)

dnn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
dnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

dnn_model.fit(X_train, y_train_encoded, validation_split=0.2, epochs=30, batch_size=32,
              callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3), EarlyStopping(patience=5)])

dnn_acc = dnn_model.evaluate(X_test, y_test_encoded, verbose=0)[1]
print(f"\nDNN Accuracy: {dnn_acc:.4f}")
