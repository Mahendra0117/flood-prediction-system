import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. GENERATE THE DATASET
def generate_data():
    np.random.seed(42)
    n_samples = 2000
    annual_rainfall = np.random.uniform(500, 3500, n_samples)
    cloud_visibility = np.random.uniform(0, 10, n_samples)
    monsoon_rainfall = annual_rainfall * np.random.uniform(0.5, 0.8, n_samples)
    
    flood_prob = (monsoon_rainfall * 0.7) - (cloud_visibility * 50)
    flood_target = (flood_prob > np.percentile(flood_prob, 70)).astype(int)
    
    df = pd.DataFrame({
        'annual_rainfall': annual_rainfall,
        'cloud_visibility': cloud_visibility,
        'monsoon_rainfall': monsoon_rainfall,
        'flood': flood_target
    })
    df.to_csv('historical_weather.csv', index=False)
    print("✓ Created 'historical_weather.csv'")

generate_data()

# 2. LOAD AND TRAIN MODEL
df = pd.read_csv('historical_weather.csv')
X = df[['annual_rainfall', 'cloud_visibility', 'monsoon_rainfall']]
y = df['flood']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# We use Random Forest since it hit 99% accuracy!
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
print(f"★ Model trained with {accuracy_score(y_test, predictions) * 100:.2f}% accuracy!")

# 3. SAVE THE ARTIFACTS
with open('flood_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("✓ Saved 'flood_model.pkl' and 'scaler.pkl'!")