import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data_path = "data/gdp_data.csv"  # Update path as needed
data = pd.read_csv(data_path)

# Display initial dataset information
print("Initial Dataset Info:")
print(data.info())

# Handle missing values
print("\nHandling missing values...")
data.replace('..', pd.NA, inplace=True)  # Replace placeholder '..' with NaN
data.fillna(method='ffill', inplace=True)  # Forward fill
data.fillna(method='bfill', inplace=True)  # Backward fill

# Ensure all columns are numeric where necessary
print("\nConverting relevant columns to numeric...")
numeric_columns = ['population_total', 'gni_atlas_usd', 'gni_per_capita_atlas_usd', 'gdp_usd', 'gdp_growth']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with missing values after handling placeholders
data.dropna(subset=numeric_columns, inplace=True)

# Handle outliers (capping extreme values)
print("\nCapping extreme outliers...")
for col in numeric_columns:
    lower_limit = data[col].quantile(0.01)
    upper_limit = data[col].quantile(0.99)
    data[col] = data[col].clip(lower=lower_limit, upper=upper_limit)

# Feature Engineering
print("\nGenerating new features...")
data['gdp_growth_rate'] = data.groupby('country_name')['gdp_usd'].pct_change() * 100
data['gni_to_gdp_ratio'] = data['gni_per_capita_atlas_usd'] / data['gdp_usd']

# Handle NaN values in newly created features
data['gdp_growth_rate'].fillna(0, inplace=True)  # Fill NaN growth rates with 0
data['gni_to_gdp_ratio'].fillna(0, inplace=True)  # Fill NaN ratios with 0

# Scaling numeric features
print("\nScaling numeric features...")
scaler = MinMaxScaler()
scaled_columns = ['population_total', 'gni_atlas_usd', 'gdp_usd', 'gdp_growth_rate', 'gni_to_gdp_ratio']
data[scaled_columns] = scaler.fit_transform(data[scaled_columns])

# Save preprocessed data
processed_path = "data/preprocessed_data.csv"
data.to_csv(processed_path, index=False)
print(f"Preprocessed data saved to {processed_path}")
data_2023 = data[data['year'] == 2023]
print(data_2023[['country_name', 'year', 'gdp_growth']])

data_path = "data/preprocessed_data.csv"
data = pd.read_csv(data_path)
print(data.head())
