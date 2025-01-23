import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import os
from sklearn.metrics import mean_squared_error, r2_score


# Load models
MODEL_PATH = "models"
rf_model, rf_features = joblib.load(os.path.join(MODEL_PATH, "random_forest_model.pkl"))
xgb_model, xgb_features = joblib.load(os.path.join(MODEL_PATH, "xgboost_model.pkl"))

# Load data
data_path = "data/preprocessed_data.csv"
data = pd.read_csv(data_path)

# Dropdown for main sections
section = st.sidebar.selectbox("Select Section", ["EDA", "GDP Analysis", "Model Prediction"])

# Section 1: EDA
if section == "EDA":
    st.header("Exploratory Data Analysis")
    st.dataframe(data.head())
    st.subheader("Correlation Matrix")
    numeric_data = data.select_dtypes(include='number')  # Select only numeric columns
    corr_matrix = numeric_data.corr()
    plt.figure(figsize=(12, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    st.pyplot(plt)


# Section 2: GDP Analysis
elif section == "GDP Analysis":
    st.header("GDP Analysis")
    analysis_type = st.radio("Select Analysis Type", 
                          ["GDP Growth of a Country", 
                           "GDP Value Comparison Between Countries", 
                           "GDP Growth Comparison Worldwide", 
                           "GDP Trends Worldwide", 
                           "Top GDP Performers"])

    # GDP Growth of a Country
    if analysis_type == "GDP Growth of a Country":
        st.subheader("Growth Rate of a Country")
        selected_country = st.sidebar.selectbox("Select Country", data['country_name'].unique())
        if selected_country:
            st.subheader(f"GDP Growth of {selected_country}")
            country_data = data[data['country_name'] == selected_country]
            country_data['growth_rate'] = country_data['gdp_usd'].pct_change() * 100
            st.line_chart(country_data.set_index('year')['growth_rate'])
            st.write(country_data)

    # GDP Value Comparison Between Countries
    elif analysis_type == "GDP Value Comparison Between Countries":
        st.subheader("GDP Value Comparison Between Countries")
        selected_countries = st.sidebar.multiselect("Select Countries", data['country_name'].unique())
        if selected_countries:
            comparison_data = data[data['country_name'].isin(selected_countries)]
            fig = px.line(comparison_data, x='year', y='gdp_usd', color='country_name', title='GDP Value Comparison')
            st.plotly_chart(fig)

    # GDP Growth Comparison Worldwide
    elif analysis_type == "GDP Growth Comparison Worldwide":
        st.subheader("GDP Growth Comparison Worldwide")
        data['gdp_growth_rate'] = data.groupby('country_name')['gdp_usd'].pct_change() * 100
        fig = px.line(data, x='year', y='gdp_growth_rate', color='country_name', title='GDP Growth Comparison Worldwide')
        st.plotly_chart(fig)

    # GDP Trends Worldwide
    elif analysis_type == "GDP Trends Worldwide":
        st.subheader("GDP Trends Worldwide")
        fig = px.line(data, x='year', y='gdp_usd', color='country_name', title='GDP Trends Worldwide')
        st.plotly_chart(fig)

    # Top GDP Performers
    elif analysis_type == "Top GDP Performers":
        st.subheader("Top GDP Performers")
        top_countries = data.groupby('country_name')['gdp_usd'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(top_countries, x=top_countries.index, y=top_countries.values, title='Top GDP Performers')
        st.plotly_chart(fig)


# Section 3: Model Prediction
elif section == "Model Prediction":
    # Displaying a table of predictions
    prediction_data = pd.read_csv("models/predictions_2023.csv")
    st.subheader('Predictions for GDP Growth (2023)')
    
    # Including actual 'gdp_growth' along with predictions
    prediction_data1 = prediction_data[['country_name', 'gdp_growth', 'random_forest_pred', 'xgboost_pred']]
    st.dataframe(prediction_data1)

    # Function to calculate RMSE
    def calculate_rmse(actual, predicted):
        mse = mean_squared_error(actual, predicted)
        rmse = mse ** 0.5  # Root Mean Squared Error
        return rmse, mse

    # Function to calculate R² (coefficient of determination)
    def calculate_r2(actual, predicted):
        return r2_score(actual, predicted)

    # Calculate accuracy for each model (using actual GDP growth data from 2023 if available)
    if 'gdp_growth' in prediction_data.columns:
        actual_values = prediction_data['gdp_growth']
        
        # Calculate RMSE, MSE, and R² for each model
        rf_rmse, rf_mse = calculate_rmse(actual_values, prediction_data['random_forest_pred'])
        xgb_rmse, xgb_mse = calculate_rmse(actual_values, prediction_data['xgboost_pred'])

        rf_r2 = calculate_r2(actual_values, prediction_data['random_forest_pred'])
        xgb_r2 = calculate_r2(actual_values, prediction_data['xgboost_pred'])

        # Display RMSE, MSE, R² values for each model
        st.subheader('Comparative Model Accuracy (RMSE, MSE, R²)')
        
        accuracy_data = {
            'Model': ['Random Forest', 'XGBoost'],
            'RMSE': [rf_rmse, xgb_rmse],
            'MSE': [rf_mse, xgb_mse],
            'R²': [rf_r2, xgb_r2],
        }
        accuracy_df = pd.DataFrame(accuracy_data)
        st.dataframe(accuracy_df)
    else:
        st.warning('Actual GDP growth data for 2023 is not available in the prediction dataset.')