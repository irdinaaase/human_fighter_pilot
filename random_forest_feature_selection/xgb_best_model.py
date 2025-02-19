import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

# Load the dataset (same one used for training)
# file_path = "STTHK3013_pilot_performance_simulation_data.xlsx"
file_path = r"C:\Users\Irdina Balqis\Documents\GitHub\human_fighter_pilot\dataset\STTHK3013_pilot_performance_simulation_data.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Feature Engineering: Add Interaction and Polynomial Features
df['reaction_stress_interaction'] = df['time_reaction'] * df['environmental_stressors']
df['fatigue_mission_ratio'] = df['fatigue_level'] / (df['mission_complexity'] + 1)
df['heart_rate_squared'] = df['heart_rate'] ** 2

# Define selected features
selected_features = ['mission_complexity', 'cognitive_level', 'stress_level', 'time_reaction', 
                     'reaction_stress_interaction', 'fatigue_mission_ratio', 'heart_rate_squared']

df.fillna(df.median(), inplace=True)  # Handle missing values
X = df[selected_features]
y = df['final_performance']

# Standardize features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Train the best XGBoost model
xgb_best = XGBClassifier(
    objective='multi:softprob', learning_rate=0.01, max_depth=12, 
    n_estimators=400, random_state=42, eval_metric='mlogloss'
)
xgb_best.fit(X_scaled, y)

# Save the model and scaler
joblib.dump(xgb_best, "xgb_best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("XGBoost Model and Scaler saved successfully!")
