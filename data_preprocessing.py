import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer


np.random.seed(42)
n_rows = 1000

raw_data = {
    'order_id': [f'ORD-{1000 + i}' for i in range(n_rows)],
    'distance_km': np.random.uniform(1.0, 45.0, n_rows),
    'package_weight_kg': np.random.uniform(0.5, 25.0, n_rows),
    'actual_delivery_min': np.random.uniform(10.0, 150.0, n_rows),
    'traffic_density': np.random.choice(['Low', 'Medium', 'High', 'Congested', None], n_rows, p=[0.25, 0.35, 0.25, 0.10, 0.05]),
    'fuel_consumed_liters': np.random.uniform(0.5, 12.0, n_rows)
}

df = pd.DataFrame(raw_data)


df.loc[15:25, 'fuel_consumed_liters'] = np.nan
df.loc[50, 'actual_delivery_min'] = 950.0  



df['traffic_density'].fillna(df['traffic_density'].mode()[0], inplace=True)


knn_imp = KNNImputer(n_neighbors=5)
df[['distance_km', 'fuel_consumed_liters']] = knn_imp.fit_transform(df[['distance_km', 'fuel_consumed_liters']])


def cap_outliers_iqr(dataframe, col_name):
    Q1 = dataframe[col_name].quantile(0.25)
    Q3 = dataframe[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    dataframe[col_name] = np.where(dataframe[col_name] > upper_bound, upper_bound,
                                  np.where(dataframe[col_name] < lower_bound, lower_bound, dataframe[col_name]))
    return dataframe

df = cap_outliers_iqr(df, 'actual_delivery_min')


scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['distance_km', 'package_weight_kg', 'actual_delivery_min', 'fuel_consumed_liters']])
scaled_df = pd.DataFrame(scaled_features, columns=['distance_scaled', 'weight_scaled', 'delivery_time_scaled', 'fuel_scaled'])


traffic_encoded = pd.get_dummies(df['traffic_density'], prefix='traffic', drop_first=True)


processed_df = pd.concat([df[['order_id']], scaled_df, traffic_encoded], axis=1)
print("Pipeline Successfully Executed. Cleaned Dataset Shape:", processed_df.shape)