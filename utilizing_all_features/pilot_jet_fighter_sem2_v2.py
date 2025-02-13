import pandas as pd
import numpy as np
import xgboost
import sklearn
print(f"xgboost version: {xgboost.__version__}")
print(f"sklearn version: {sklearn.__version__}")

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost.sklearn import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load the dataset
file_path = 'STTHK3013_pilot_performance_simulation_data.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')
print(f"\nData shape: {data.shape}")

# Handle missing values
data.dropna(subset=['final_performance'], inplace=True)
continuous_columns = [
    'heart_rate', 'sleep_quality', 'mission_complexity', 'experience_level',
    'environmental_stressors', 'cognitive_level', 'fatigue_level', 'stress_level', 'time_reaction'
]
for col in continuous_columns:
    mean_value = data[col].mean()
    data[col].fillna(mean_value, inplace=True)

# Feature Engineering: Creating new features
data["fatigue_stress_index"] = data["fatigue_level"] * data["stress_level"]
data["cognitive_experience_score"] = data["cognitive_level"] * data["experience_level"]
data["adaptability_score"] = data["experience_level"] / (data["environmental_stressors"] + data["mission_complexity"] + 1)
data["normalized_reaction_time"] = data["time_reaction"] / data["time_reaction"].max()

# Encode the target variable if it's categorical
if data['final_performance'].dtype == 'object':
    label_encoder = LabelEncoder()
    data['final_performance'] = label_encoder.fit_transform(data['final_performance'])

# Define features (X) and target (y)
X = data.drop(columns=['final_performance'])
y = data['final_performance']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Print class distribution before SMOTE
print("\nOriginal Class Distribution:", Counter(y))

# Apply SMOTE with sampling strategy
sampling_strategy = {
    0: 169,  # Match Class 0 to Class 1 size
    1: 181,  # Match Class 1 to Class 2 size
    2: 185,  # Match Class 2 to Class 3 size
    3: 188,  # Match Class 3 to Class 4 size
    4: 203,  # Class 4 will remain as is (largest)
    5: 176   # Match Class 5 to Class 1 size
}
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=100)

log_reg_accuracies = []
xgb_accuracies = []
dnn_accuracies = []

# Loop through each fold
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Apply SMOTE only to the training set
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Check class distribution after SMOTE
    print(f"Class Distribution After SMOTE (Train): {Counter(y_train_smote)}")

    # Logistic Regression Model
    log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
    log_reg_model.fit(X_train_smote, y_train_smote)
    y_pred_log_reg = log_reg_model.predict(X_test)
    log_reg_accuracies.append(accuracy_score(y_test, y_pred_log_reg))

    # XGBoost Model with GridSearchCV
    xgb_model = XGBClassifier(random_state=42, eval_metric='mlogloss')  # Add eval_metric

    # Define parameter grid for GridSearchCV
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Perform GridSearchCV
    xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=3, n_jobs=-1)
    xgb_grid_search.fit(X_train_smote, y_train_smote)

    # Print the best parameters and model
    print(f"\nBest Parameters from GridSearchCV: {xgb_grid_search.best_params_}")
    print(f"Best XGBoost Model: {xgb_grid_search.best_estimator_}")

    # Get the best model and evaluate
    best_xgb_model = xgb_grid_search.best_estimator_
    y_pred_xgb = best_xgb_model.predict(X_test)
    xgb_accuracies.append(accuracy_score(y_test, y_pred_xgb))

    # Deep Neural Network Model (DNN)
    y_train_encoded = to_categorical(y_train_smote, num_classes=len(y.unique()))
    y_test_encoded = to_categorical(y_test, num_classes=len(y.unique()))

    dnn_model = Sequential([
        Dense(64, input_dim=X_train_smote.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(y.unique()), activation='softmax')
    ])
    dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    dnn_model.fit(X_train_smote, y_train_encoded, epochs=10, batch_size=32, verbose=0)
    _, dnn_accuracy = dnn_model.evaluate(X_test, y_test_encoded, verbose=0)
    dnn_accuracies.append(dnn_accuracy)

# Calculate and print the average accuracy for each model
print("\nModel Performance Summary:")
print("Logistic Regression Average Accuracy:", np.mean(log_reg_accuracies))
print("XGBoost Average Accuracy:", np.mean(xgb_accuracies))
print("DNN Average Accuracy:", np.mean(dnn_accuracies))
