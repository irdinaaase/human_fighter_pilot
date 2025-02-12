import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'STTHK3013_pilot_performance_simulation_data.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# Handle missing values
continuous_columns = [
    'heart_rate', 'sleep_quality', 'mission_complexity', 'experience_level', 
    'environmental_stressors', 'cognitive_level', 'fatigue_level', 'stress_level', 'time_reaction'
]
for col in continuous_columns:
    mean_value = data[col].mean()
    data[col].fillna(mean_value, inplace=True)

data.dropna(subset=['final_performance'], inplace=True)

# Define features (X) and target (y)
X = data.drop(columns=['final_performance'])
y = data['final_performance']

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply Borderline-SMOTE for oversampling
borderline_smote = BorderlineSMOTE(random_state=42, kind="borderline-1")
X_train_resampled, y_train_resampled = borderline_smote.fit_resample(X_train, y_train)

# ------------------ Hyperparameter Tuning for Logistic Regression ------------------
param_dist_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['saga', 'liblinear']
}
lr_tuned = RandomizedSearchCV(
    LogisticRegression(random_state=42, max_iter=500), 
    param_distributions=param_dist_lr, 
    cv=5, 
    scoring='f1_weighted', 
    n_iter=20, 
    n_jobs=-1, 
    random_state=42
)
lr_tuned.fit(X_train_resampled, y_train_resampled)
best_lr_model = lr_tuned.best_estimator_

# Evaluate the tuned Logistic Regression model
y_pred_lr = best_lr_model.predict(X_test)
y_prob_lr = best_lr_model.predict_proba(X_test)[:, 1]  # For ROC curve
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr, average='weighted')

# ------------------ Hyperparameter Tuning for XGBoost ------------------
param_dist_xgb = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [4, 6, 8, 10],
    'n_estimators': [100, 200, 300, 400],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
xgb_tuned = RandomizedSearchCV(
    XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), 
    param_distributions=param_dist_xgb, 
    cv=5, 
    scoring='f1_weighted', 
    n_iter=20, 
    n_jobs=-1, 
    random_state=42
)
xgb_tuned.fit(X_train_resampled, y_train_resampled)
best_xgb_model = xgb_tuned.best_estimator_

# Evaluate the tuned XGBoost model
y_pred_xgb = best_xgb_model.predict(X_test)
y_prob_xgb = best_xgb_model.predict_proba(X_test)[:, 1]  # For ROC curve
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb, average='weighted')

# ------------------ MLP Model Creation ------------------

# Create the MLP model using the Keras wrapper
def create_mlp_model(hidden_units=512):
    model = Sequential([
        Dense(hidden_units, input_dim=X_train.shape[1], activation='relu'),
        Dense(hidden_units//2, activation='relu'),
        Dense(hidden_units//4, activation='relu'),
        Dense(y_train_encoded.shape[1], activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ------------------ GridSearchCV for MLP ------------------

# One-hot encode the target variable
y_train_encoded = to_categorical(y_train_resampled)
y_test_encoded = to_categorical(y_test)

# Hyperparameter grid for MLP
param_grid_mlp = {
    'hidden_units': [128, 256, 512],
    'epochs': [50, 100, 150],
    'batch_size': [32, 64]
}

# Use KerasClassifier to wrap the MLP model
mlp_model = KerasClassifier(build_fn=create_mlp_model, verbose=0)

# Use GridSearchCV to tune the hyperparameters
grid_search_mlp = GridSearchCV(
    estimator=mlp_model,
    param_grid=param_grid_mlp,
    n_jobs=-1,
    cv=3,
    verbose=1,
    scoring='accuracy'
)

# Perform GridSearch on MLP
grid_search_mlp.fit(X_train_resampled, y_train_encoded)

# Get the best model from GridSearch
best_mlp_params = grid_search_mlp.best_params_

# Create and train the best model
best_mlp_model = create_mlp_model(hidden_units=best_mlp_params['hidden_units'])
best_mlp_model.fit(X_train_resampled, y_train_encoded, epochs=best_mlp_params['epochs'], batch_size=best_mlp_params['batch_size'])

# Evaluate the best MLP model
loss, accuracy = best_mlp_model.evaluate(X_test, y_test_encoded, verbose=0)
y_pred_mlp = np.argmax(best_mlp_model.predict(X_test), axis=1)
mlp_accuracy = accuracy
mlp_f1 = f1_score(np.argmax(y_test_encoded, axis=1), y_pred_mlp, average='weighted')

# ------------------ ROC Curves ------------------
plt.figure(figsize=(10, 8))

# Logistic Regression ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')

# XGBoost ROC
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')

# MLP ROC
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, best_mlp_model.predict(X_test)[:, 1])
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)
plt.plot(fpr_mlp, tpr_mlp, label=f'MLP (AUC = {roc_auc_mlp:.2f})')

plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# ------------------ Summary of Results ------------------
results = pd.DataFrame({
    "Model": ["Logistic Regression", "XGBoost", "MLP"],
    "Accuracy": [lr_accuracy, xgb_accuracy, mlp_accuracy],
    "F1-Score": [lr_f1, xgb_f1, mlp_f1]
})
print("\nSummary of Model Performance:")
print(results)
