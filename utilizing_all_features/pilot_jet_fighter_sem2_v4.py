# -------------------- IMPORT LIBRARIES --------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
import tensorflow as tf
import joblib
import keras
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter

# -------------------- LOAD DATA --------------------
file_path = r"C:\Users\Irdina Balqis\Documents\GitHub\human_fighter_pilot\dataset\STTHK3013_pilot_performance_simulation_data.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Remove rows with missing target values
df.dropna(subset=['final_performance'], inplace=True)

# -------------------- DATA PREPROCESSING --------------------
def categorize_performance(value):
    if value <= 1:
        return 0  # Low
    elif 2 <= value <= 3:
        return 1  # Medium
    elif 4 <= value <= 5:
        return 2  # High

df['final_performance'] = df['final_performance'].apply(categorize_performance)

# Define target and features
target_column = "final_performance"
X = df.drop(columns=[target_column])
y = df[target_column]

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Keep the data as DataFrame for easier manipulation
knn_imputer = KNNImputer(n_neighbors=5)
X_train = pd.DataFrame(knn_imputer.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(knn_imputer.transform(X_test), columns=X.columns)  # Avoid leakage

# -------------------- FEATURE ENGINEERING --------------------
# Feature Engineer using pandas DataFrame (now that we have X_train as DataFrame)
X_train["fatigue_stress_index"] = X_train["fatigue_level"] * X_train["stress_level"]
X_train["cognitive_experience_score"] = X_train["cognitive_level"] * X_train["experience_level"]
X_train["adaptability_score"] = X_train["experience_level"] / (X_train["environmental_stressors"] + X_train["mission_complexity"] + 1)
X_train["normalized_reaction_time"] = np.log1p(X_train["time_reaction"] / X_train["time_reaction"].max())

X_test["fatigue_stress_index"] = X_test["fatigue_level"] * X_test["stress_level"]
X_test["cognitive_experience_score"] = X_test["cognitive_level"] * X_test["experience_level"]
X_test["adaptability_score"] = X_test["experience_level"] / (X_test["environmental_stressors"] + X_test["mission_complexity"] + 1)
X_test["normalized_reaction_time"] = np.log1p(X_test["time_reaction"] / X_test["time_reaction"].max())

# # Drop redundant features
# drop_cols = ['fatigue_level', 'stress_level', 'cognitive_level', 'experience_level', 'environmental_stressors', 'mission_complexity', 'time_reaction']
# X_train.drop(columns=drop_cols, inplace=True)
# X_test.drop(columns=drop_cols, inplace=True)

# Handle imbalanced data with SMOTE
print("\nClass Distribution Before SMOTE:", Counter(y_train))

smote = SMOTE(sampling_strategy={2: 286, 1: 269, 0: 274}, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("\nClass Distribution After SMOTE:", Counter(y_train_smote))

# Feature Interaction
X_train_smote["reaction_fatigue_stress"] = X_train_smote["normalized_reaction_time"] * X_train_smote["fatigue_stress_index"]
X_train_smote["cognitive_adaptability"] = X_train_smote["cognitive_experience_score"] / (X_train_smote["adaptability_score"] + 1e-6)
X_train_smote["reaction_experience"] = X_train_smote["normalized_reaction_time"] * X_train_smote["cognitive_experience_score"]

X_test["reaction_fatigue_stress"] = X_test["normalized_reaction_time"] * X_test["fatigue_stress_index"]
X_test["cognitive_adaptability"] = X_test["cognitive_experience_score"] / (X_test["adaptability_score"] + 1e-6)
X_test["reaction_experience"] = X_test["normalized_reaction_time"] * X_test["cognitive_experience_score"]

# Polynomial Feature Interactions
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_smote)
X_test_poly = poly.transform(X_test)

X_train_smote = pd.DataFrame(X_train_poly, columns=poly.get_feature_names_out(X_train_smote.columns))
X_test = pd.DataFrame(X_test_poly, columns=poly.get_feature_names_out(X_test.columns))

# -------------------- DOMAIN KNOWLEDGE FEATURE BINNING --------------------
def categorize_sleep_quality(value):
    if value <= 2:
        return "poor"
    elif 3 <= value <= 4:
        return "average"
    else:
        return "good"

def categorize_experience(value):
    if value <= 2:
        return "low"
    elif 3 <= value <= 5:
        return "medium"
    else:
        return "high"

# Apply categorization
X_train_smote["sleep_category"] = X_train_smote["sleep_quality"].apply(categorize_sleep_quality)
X_train_smote["experience_category"] = X_train_smote["experience_level"].apply(categorize_experience)

X_test["sleep_category"] = X_test["sleep_quality"].apply(categorize_sleep_quality)
X_test["experience_category"] = X_test["experience_level"].apply(categorize_experience)

# One-hot encoding
X_train_smote = pd.get_dummies(X_train_smote, columns=["sleep_category", "experience_category"], drop_first=True)
X_test = pd.get_dummies(X_test, columns=["sleep_category", "experience_category"], drop_first=True)

# Drop 'experience_level' only after it's no longer needed
X_train_smote.drop(columns=['experience_level'], inplace=True)
X_test.drop(columns=['experience_level'], inplace=True)

# -------------------- FEATURE WEIGHTING USING XGBOOST --------------------
xgb_model = XGBClassifier(random_state=42, eval_metric="mlogloss")
xgb_model.fit(X_train_smote, y_train_smote)

# Get feature importance
importance = xgb_model.feature_importances_

# Scale feature importance to range (0.5, 1)
importance_scaled = MinMaxScaler(feature_range=(0.5, 1)).fit_transform(importance.reshape(-1, 1)).flatten()

# Apply feature weights to dataset
for i, col in enumerate(X_train_smote.columns):
    X_train_smote[col] *= importance_scaled[i]
    X_test[col] *= importance_scaled[i]

# -------------------- K-FOLD CROSS-VALIDATION --------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

log_reg_accuracies = []
xgb_accuracies = []
dnn_accuracies = []

log_reg_param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga']}

# -------------------- MODEL TRAINING & TESTING --------------------
for train_index, test_index in kf.split(X_train_smote, y_train_smote):
    X_k_train, X_k_test = X_train_smote.iloc[train_index], X_train_smote.iloc[test_index]
    y_k_train, y_k_test = y_train_smote.iloc[train_index], y_train_smote.iloc[test_index]

    # -------------------- LOGISTIC REGRESSION --------------------
    log_reg_model = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                                log_reg_param_grid, cv=3, scoring='accuracy', verbose=3)
    log_reg_model.fit(X_k_train, y_k_train)
    y_pred_log_reg = log_reg_model.best_estimator_.predict(X_k_test)
    log_reg_accuracies.append(accuracy_score(y_k_test, y_pred_log_reg))

    # -------------------- XGBOOST --------------------
    xgb_param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],  
        'max_depth': [3, 5, 7, 9],  
        'n_estimators': [100, 200, 500],  
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],  # Regularization
        'reg_lambda': [0, 1, 10],  # L2 Regularization
    }
    
    xgb_model = GridSearchCV(XGBClassifier(random_state=42, eval_metric='mlogloss'),
                         param_grid=xgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=3)
    xgb_model.fit(X_k_train, y_k_train)
    y_pred_xgb = xgb_model.best_estimator_.predict(X_k_test)
    xgb_accuracies.append(accuracy_score(y_k_test, y_pred_xgb))

    # -------------------- DEEP NEURAL NETWORK (DNN) --------------------
    num_classes = len(np.unique(y_train_smote))

    y_k_train_encoded = to_categorical(y_k_train, num_classes=num_classes)
    y_k_test_encoded = to_categorical(y_k_test, num_classes=num_classes)

    dnn_model = Sequential([
        Dense(1024, activation='relu', kernel_regularizer=l2(0.01), input_dim=X_train.shape[1]),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')  # Multi-class classification
    ])
    
    # Learning rate decay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9)

    # Compile model
    dnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                    loss='categorical_crossentropy', metrics=['accuracy'])  
    
    # Print model summary
    print("\n✅ DNN Model Summary:")
    dnn_model.summary()

    # Callback to print loss & accuracy at each epoch
    epoch_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch + 1}: Loss = {logs['loss']:.4f}, Accuracy = {logs['accuracy']:.4f}")
    )

    # Train model
    history = dnn_model.fit(X_k_train, y_k_train_encoded, validation_split=0.2, epochs=50, batch_size=32, 
                            verbose=1, callbacks=[epoch_callback])

    _, dnn_accuracy = dnn_model.evaluate(X_k_test, y_k_test_encoded)
    dnn_accuracies.append(dnn_accuracy)

# -------------------- PRINT FINAL RESULTS --------------------
print("\nModel Performance Summary:")

print(f"Logistic Regression Average Accuracy: {np.mean(log_reg_accuracies):.4f}")
best_log_reg_model = log_reg_model.best_estimator_

print(f"XGBoost Average Accuracy: {np.mean(xgb_accuracies):.4f}")
best_xgb_model = xgb_model.best_estimator_

joblib.dump(best_xgb_model, "xgb_best_model.pkl")
print("✅ XGBoost model saved as xgb_best_model.pkl")

print(f"DNN Average Accuracy: {np.mean(dnn_accuracies):.4f}")
best_dnn_model = dnn_model

# -------------------- VISUALIZATIONS --------------------
# Model performance comparison
model_names = ['Logistic Regression', 'XGBoost', 'DNN']
accuracies = [np.mean(log_reg_accuracies), np.mean(xgb_accuracies), np.mean(dnn_accuracies)]

plt.figure(figsize=(8, 5))
sns.barplot(x=model_names, y=accuracies)
plt.title("Average Model Accuracies")
plt.ylabel("Accuracy")
plt.show()

# Confusion matrix for the best model (XGBoost)
y_pred_best_model = best_xgb_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best_model)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.title("Confusion Matrix - Best XGBoost Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Confusion matrix for the best model (Logistic Regression)
y_pred_best_model = best_log_reg_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best_model)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.title("Confusion Matrix - Best Logistic Regression Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Confusion matrix for the best model (DNN)
y_pred_best_model = np.argmax(best_dnn_model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred_best_model)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.title("Confusion Matrix - Best DNN Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
