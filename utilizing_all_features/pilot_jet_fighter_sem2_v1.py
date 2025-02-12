# Model Comparison:
# Logistic Regression Accuracy: 0.21621621621621623
# XGBoost Accuracy: 0.13513513513513514
# Deep Neural Network Accuracy: 0.23783783614635468

import tensorflow as tf
print(tf.__version__)  # Should print 2.18.0
import keras
print(keras.__version__)  # Should print 3.8.0

# Import necessary libraries
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

# Load the dataset from the specified Excel file
file_path = 'STTHK3013_pilot_performance_simulation_data.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')
print(f"\ndata shape: {data.shape}")

# Analyze and plot the distribution of the target variable 'final_performance'
target_distribution = data['final_performance'].value_counts().sort_index()

# # Plot the distribution of 'final_performance'
# plt.figure(figsize=(10, 6))
# sns.barplot(x=target_distribution.index, y=target_distribution.values, palette="viridis")
# plt.title('Distribution of Final Performance', fontsize=16)
# plt.xlabel('Final Performance', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()

# # Print the distribution of target variable
# print("Final Performance Distribution:\n", target_distribution)

# # Display the shape of the cleaned data (number of rows before and after)
# print(f"\ndata shape: {data.shape}")

# # Check for missing values in the dataset
# missing_values = data.isnull().sum()
# print ("\nMissing Values Summary:\n", missing_values.sum(), "missing values in the dataset")
# print("\nMissing Values per Feature:\n", missing_values[missing_values > 0])

# # Visualize missing values as a heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
# plt.title('Missing Values Heatmap', fontsize=16)
# plt.show()

# Drop rows where 'final_performance' is missing
data.dropna(subset=['final_performance'], inplace=True)

# Impute continuous columns with mean
continuous_columns = [
    'heart_rate', 'sleep_quality', 'mission_complexity', 'experience_level', 
    'environmental_stressors', 'cognitive_level', 'fatigue_level', 'stress_level', 'time_reaction'
]

for col in continuous_columns:
    mean_value = data[col].mean()
    data[col].fillna(mean_value, inplace=True)

# Verify that missing values are handled
# print("Missing Values After Imputation:")
# print(data.isnull().sum())

# # Display the first few rows of the dataset after imputation
# print(data.head())

# print(f"\ndata shape after minus final perfromance: {data.shape}") 

# # Check the remaining missing values in the dataset
# missing_values_after_drop = data.isnull().sum()
# print("\nMissing Values After Dropping Rows:\n", missing_values_after_drop.sum(), "missing values left in the dataset")
# print("\nMissing Values Per Feature After Drop:\n", missing_values_after_drop[missing_values_after_drop > 0])

# # Display the first few rows of the dataset after imputation
# print(data.head())

# # Visualize missing values as a heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
# plt.title('Missing Values Heatmap', fontsize=16)
# plt.show()

# # Plot distribution of each variable
# columns = data.columns
# plt.figure(figsize=(16, 20))

# print(f"\ndata shape: {data.shape}") 

# Adjust subplot layout dynamically for better spacing
# plt.figure(figsize=(15, len(columns) * 2))  # Adjust figure size based on the number of columns

# for i, col in enumerate(columns, 1):
#     plt.subplot((len(columns) + 2) // 3, 3, i)  # Arrange plots in a grid
#     if data[col].nunique() < 20:  # Discrete/categorical
#         sns.countplot(x=col, data=data, palette="viridis")  # Explicitly set x and data
#         plt.title(f'Distribution of {col} (Categorical)', fontsize=12)
#     else:  # Continuous
#         sns.histplot(data[col], kde=True, bins=30, color='blue')
#         plt.title(f'Distribution of {col} (Continuous)', fontsize=12)
    
#     plt.xlabel(col, fontsize=10)
#     plt.ylabel('Frequency', fontsize=10)
#     plt.xticks(fontsize=8)
#     plt.yticks(fontsize=8)

# plt.tight_layout()
# plt.show()
# print(f"\ndata shape: {data.shape}") 

# Outlier detection using IQR for numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns

outlier_summary = {}  # Dictionary to store the count of outliers for each column
total_outliers = 0  # Initialize total outliers counter

# Set up the grid for subplots
num_cols = len(numeric_cols)
nrows = (num_cols + 2) // 3  # Number of rows (3 plots per row)
fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, nrows * 5))  # Adjust figure size
axes = axes.flatten()  # Flatten the axes array for easy indexing

# Initialize a mask for rows to keep (True for rows to keep)
rows_to_keep = np.ones(len(data), dtype=bool)

for i, col in enumerate(numeric_cols):
    # Calculate Q1, Q3, and IQR
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_summary[col] = outlier_count  # Store count in the dictionary

    # Print the number of outliers in this column
    print(f"Number of outliers in {col}: {outlier_count}")
    
    # Print the outliers for this column
    print(f"\nOutliers in {col}:\n", outliers)

    # Count the number of outliers
    num_outliers = outliers.shape[0]
    total_outliers += num_outliers  # Add to the total count
    print(f"Number of outliers in {col}: {num_outliers}")

    # Update the rows_to_keep mask (keep rows that are not outliers)
    rows_to_keep = rows_to_keep & ~((data[col] < lower_bound) | (data[col] > upper_bound))

    # Plot boxplot in the respective subplot
    sns.boxplot(x=data[col], color='orange', ax=axes[i])
    axes[i].set_title(f'Outliers in {col}', fontsize=14)
    axes[i].set_xlabel(col, fontsize=12)
    axes[i].tick_params(axis='x', labelsize=10)

# Hide any unused subplots (if there are fewer columns than subplots)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# # Adjust layout and display all plots on one page
# plt.tight_layout()
# plt.show()

# Display a summary of outliers
print("\nSummary of outliers by column:")
for col, count in outlier_summary.items():
    print(f"{col}: {count} outliers")

# Print total number of outliers
print(f"\nTotal number of outliers in all columns: {total_outliers}")
print(f"\ndata shape: {data.shape}") 

# Drop rows with outliers based on the mask
data_cleaned = data[rows_to_keep]

# Display the shape of the cleaned data (number of rows before and after)
print(f"\nOriginal data shape: {data.shape}")
print(f"Cleaned data shape: {data_cleaned.shape}")

# Compute the correlation matrix
correlation_matrix = data_cleaned.corr()

# # Plot the correlation matrix
# plt.figure(figsize=(12, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
# plt.title('Correlation Matrix')
# plt.show()

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data['feature'] = data_cleaned.columns
vif_data['VIF'] = [variance_inflation_factor(data_cleaned.values, i) for i in range(data_cleaned.shape[1])]

# Display VIF for each feature
print(vif_data)

print(f"\ndata shape: {data_cleaned.shape}") 

# Calculate correlation matrix
corr_matrix = data_cleaned.corr()

# # Plot the heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
# plt.title('Correlation Heatmap', fontsize=16)
# plt.show()

print(f"\ndata shape: {data_cleaned.shape}") 

from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = data_cleaned.drop(columns=['final_performance'])  # Features
y = data_cleaned['final_performance']  # Target

print("Original Class Distribution:", Counter(y))

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Print class distribution before SMOTE
print("Original Training Class Distribution:", Counter(y_train))
print("Original Testing Class Distribution:", Counter(y_test))

# Apply SMOTE only on the training data
sampling_strategy = {
    0: 169,  # Match Class 0 to Class 1 size
    1: 181,  # Match Class 1 to Class 2 size
    2: 185,  # Match Class 2 to Class 3 size
    3: 188,  # Match Class 3 to Class 4 size
    4: 203,  # Class 4 will remain as is (largest)
    5: 176   # Match Class 5 to Class 1 size
}

smote = SMOTE(sampling_strategy=sampling_strategy, random_state=45)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Print class distribution after SMOTE
print("Training Class Distribution After SMOTE:", Counter(y_train_smote))

# Check the original class distribution

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Initialize and train the Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_train, y_train)

# Predict on the test set
y_pred_log_reg = log_reg_model.predict(X_test)

# Evaluate the model
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("Classification Report:\n", classification_report(y_test, y_pred_log_reg))


from xgboost import XGBClassifier

# Initialize and train the XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# One-hot encode the target variable
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Build the DNN model
dnn_model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(y_train_encoded.shape[1], activation='softmax')
])

# Compile the model
dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
dnn_model.fit(X_train, y_train_encoded, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
loss, accuracy = dnn_model.evaluate(X_test, y_test_encoded, verbose=0)
print("DNN Accuracy:", accuracy)

print("Model Comparison:")
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Deep Neural Network Accuracy:", accuracy)