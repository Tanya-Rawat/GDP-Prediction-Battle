# GDP Growth Prediction for 2023 and GDP Analysis

## Project Overview
This project focuses on predicting the GDP growth for countries in 2023 using machine learning models. In addition to predictions, the project provides a comprehensive analysis of historical economic data and GDP trends. The goal is to compare the performance of two regression models: **Random Forest** and **XGBoost**. The models are trained and evaluated on a dataset of historical economic indicators such as population growth, GNI, and GDP data. The project also includes accuracy analysis through metrics like **Root Mean Squared Error (RMSE)** and **Mean Squared Error (MSE)** to assess the prediction performance.

## Features
- **GDP Growth Predictions**: Predicts the GDP growth for 2023 for multiple countries using two models.
- **Interactive Analysis**: Provides tools for exploring historical trends, GDP comparisons, and top-performing countries.
- **GDP Analysis**: Offers detailed insights into GDP growth of individual countries, worldwide trends, and comparative performance.
- **Model Accuracy Comparison**: Compares model performance based on RMSE and MSE.
- **Visualizations**: Interactive charts for GDP trends, growth comparisons, and country-specific insights.

## Requirements
Make sure you have the following libraries installed:
- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- streamlit
- matplotlib
- seaborn
- plotly

To install the dependencies, run:
```bash
pip install -r requirements.txt
```
## Dataset
The dataset used in this project includes the following columns:

- `country_name`: Name of the country
- `gdp_usd`: GDP value in USD
- `gdp_growth`: Actual GDP growth for the year 2023
- `random_forest_pred`: Predicted GDP growth using Random Forest
- `xgboost_pred`: Predicted GDP growth using XGBoost

Additional historical indicators include:

- `population_total`: Total population of the country
- `population_growth`: Annual population growth rate
- `gni_atlas_usd`: Gross National Income (GNI) in USD
- `gni_per_capita_atlas_usd`: GNI per capita in USD
- `life_expectancy`: Average life expectancy
- `school_enrollment_secondary`: Percentage of secondary school enrollment

## Project Structure
```
├── models/
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── predictions_2023.csv       # Contains predictions and actual GDP data
├── data/
│   └── preprocessed_data.csv      # Preprocessed dataset
├── src/
│   └── data_preprocessing.py      # Preprocessing dataset
│   └── models.py                  # Models file to create .pkl file for all three models
├── main.py                        # Main Streamlit app
├── requirements.txt               # List of required Python libraries
└── README.md                      # Project documentation
```

## How to Run the Project
1. Clone the repository to your local machine.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the Streamlit app with the following command:
   ```bash
   streamlit run main.py
   ```
4. The application will open in your default web browser. You can:
   - Explore GDP trends and growth rates worldwide.
   - Compare GDP values across countries.
   - Analyze the predictions for GDP growth in 2023.

## Model Evaluation and Results Interpretation

### Performance Metrics:
- **Root Mean Squared Error (RMSE)**: 
  - RMSE is a commonly used metric to evaluate the performance of regression models. It calculates the square root of the average squared differences between predicted and actual values. Lower RMSE values indicate better model performance, as they signify smaller deviations between predicted and actual outcomes.

- **Mean Squared Error (MSE)**:
  - MSE is similar to RMSE but it does not take the square root. It calculates the average of the squared differences between predicted and actual values. Like RMSE, lower MSE values indicate better model performance. MSE gives more weight to larger errors, penalizing larger deviations more heavily.

### Model Performance Results

| Model             | RMSE   | MSE     |
|-------------------|--------|---------|
| Random Forest     | 2.6673 | 7.1146  |
| XGBoost           | 2.5055 | 6.2778  |

### Results Interpretation:

1. **Random Forest**:
   - The Random Forest model has an RMSE of **2.6673** and an MSE of **7.1146**. This means that, on average, the predictions made by the model are off by about 2.67% in terms of GDP growth prediction. The MSE value indicates that the model's errors are penalized more for larger deviations.
   
2. **XGBoost**:
   - The XGBoost model shows slightly better performance with an RMSE of **2.5055** and an MSE of **6.2778**. This suggests that the model's predictions are more accurate than the Random Forest model by approximately 0.16% in terms of RMSE. The MSE value also suggests that XGBoost's predictions have smaller errors and are penalized less for larger deviations.

### Interpretation:
- **XGBoost** outperforms **Random Forest** in both RMSE and MSE, indicating that XGBoost provides more accurate predictions for GDP growth in 2023. This suggests that XGBoost's ability to handle complex data patterns is more effective in this context than the Random Forest model. 
- The difference in RMSE and MSE between the two models, though modest, suggests that tuning XGBoost or exploring further optimizations (e.g., hyperparameter tuning) could lead to even better results.

In conclusion, based on the performance metrics, **XGBoost** is the preferred model for predicting GDP growth, as it demonstrates higher accuracy and smaller errors in comparison to **Random Forest**.

---
By comparing different machine learning models using these performance metrics, we can gain valuable insights into which model best suits the task of predicting GDP growth.
 project to analyze and predict GDP growth with interactive visualizations and robust machine learning models.
