import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Load the dataset (same one used for training)
file_path = "STTHK3013_pilot_performance_simulation_data.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Define selected features
selected_features = ['time_reaction', 'fatigue_level', 'heart_rate', 'mission_complexity', 'environmental_stressors', 'cognitive_level']

df.fillna(df.median(), inplace=True)  # Handle missing values
X = df[selected_features]
y = df['final_performance']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the best MLP model
mlp_best = MLPClassifier(
    activation='tanh', hidden_layer_sizes=(256, 128), learning_rate_init=0.001,
    max_iter=500, solver='adam', random_state=42
)
mlp_best.fit(X_scaled, y)

# Save the model and scaler
joblib.dump(mlp_best, "mlp_best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("MLP Model and Scaler saved successfully!")
