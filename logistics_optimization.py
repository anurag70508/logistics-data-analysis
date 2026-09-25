import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

np.random.seed(42)
data_size = 1000
logistics_data = pd.DataFrame({
    'order_id': range(1, data_size + 1),
    'latitude': np.random.uniform(40.5, 40.9, data_size),   # Simulated city coordinates
    'longitude': np.random.uniform(-74.3, -73.9, data_size),
    'distance_km': np.random.uniform(1.0, 30.0, data_size),
    'package_weight_kg': np.random.uniform(0.5, 20.0, data_size),
    'traffic_index': np.random.uniform(1.0, 5.0, data_size), # 1 = Light, 5 = Heavy
    'actual_delivery_time_min': np.random.uniform(15.0, 120.0, data_size)
})


coords = logistics_data[['latitude', 'longitude']]
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
logistics_data['delivery_zone'] = kmeans.fit_predict(coords)

print("Packages assigned per delivery zone:")
print(logistics_data['delivery_zone'].value_counts())

# 3. Regression: Predicting Delivery Time based on route features
# Features: Distance, Weight, Traffic, and Delivery Zone
X = logistics_data[['distance_km', 'package_weight_kg', 'traffic_index', 'delivery_zone']]
y = logistics_data['actual_delivery_time_min']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


predictions = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"\nModel Mean Absolute Error: {mae:.2f} minutes")
print("\nSample Prediction for a new route:")
sample_route = pd.DataFrame({'distance_km': [12.5], 'package_weight_kg': [5.0], 
                             'traffic_index': [3.2], 'delivery_zone': [2]})
predicted_time = rf_model.predict(sample_route)
print(f"Estimated Delivery Time: {predicted_time[0]:.2f} minutes")