import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Update the train_models function in models.py
def train_models(data):
    # Define the features to use for training (excluding gdp_usd)
    feature_columns = ['population_total', 'population_growth', 'gni_atlas_usd', 'gni_per_capita_atlas_usd', 'life_expectancy', 'school_enrollment_secondary', 'power_consumption_per_capita', 
                       'inflation_rate', 'agriculture_value_added', 'industry_value_added', 'exports_gdp_percent', 'imports_gdp_percent', 'capital_formation_gdp_percent', 'fdi_net_inflows_usd']
    target_column = 'gdp_growth'

    # Filter the data for training (1960 to 2022) and prediction for 2023
    training_data = data[data['year'] < 2023]
    prediction_data = data[data['year'] == 2023]
    
    # Separate features and target for training
    X_train = training_data[feature_columns]
    y_train = training_data[target_column]
    
    # Separate features for prediction (2023)
    X_pred = prediction_data[feature_columns]

    # Split the training dataset into train and test sets
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Train Random Forest
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train_split, y_train_split)

    # Save Random Forest Model
    joblib.dump((rf_model, feature_columns), "models/random_forest_model.pkl")

    # Train XGBoost
    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(X_train_split, y_train_split)

    # Save XGBoost Model
    joblib.dump((xgb_model, feature_columns), "models/xgboost_model.pkl")
    
    # Make Predictions for 2023
    rf_pred = rf_model.predict(X_pred)
    xgb_pred = xgb_model.predict(X_pred)

    # Add predictions to the 2023 data
    prediction_data = prediction_data.copy()

    # Now, you can safely assign values to the new columns
    prediction_data.loc[:, 'random_forest_pred'] = rf_pred
    prediction_data.loc[:, 'xgboost_pred'] = xgb_pred

    # Include the actual GDP growth data for 2023
    prediction_data['gdp_growth'] = prediction_data[target_column]

    # Save prediction data for 2023
    prediction_data.to_csv('models/predictions_2023.csv', index=False)


if __name__ == "__main__":
    # Load and preprocess data
    data_path = "data/preprocessed_data.csv"
    data = pd.read_csv(data_path)
    
    # Call the function to train the models
    train_models(data)
