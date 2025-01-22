# GDP Growth Prediction for 2023 and GDP Analysis

## Project Overview
This project , focuses on predicting the GDP growth for countries in 2023 using machine learning models. In addition to predictions, the project provides a comprehensive analysis of historical economic data and GDP trends. The goal is to compare the performance of three regression models: **Linear Regression**, **Random Forest**, and **XGBoost**. The models are trained and evaluated on a dataset of historical economic indicators such as population growth, GNI, and GDP data. The project also includes accuracy analysis through metrics like **Root Mean Squared Error (RMSE)** and **Accuracy Percentage** to assess the prediction performance.

## Features
- **GDP Growth Predictions**: Predicts the GDP growth for 2023 for multiple countries using three models.
- **Interactive Analysis**: Provides tools for exploring historical trends, GDP comparisons, and top-performing countries.
- **GDP Analysis**: Offers detailed insights into GDP growth of individual countries, worldwide trends, and comparative performance.
- **Model Accuracy Comparison**: Compares model performance based on RMSE and accuracy percentage.
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
- `linear_regression_pred`: Predicted GDP growth using Linear Regression
- `random_forest_pred`: Predicted GDP growth using Random Forest
- `xgboost_pred`: Predicted GDP growth using XGBoost

Additional historical indicators include:
- `population_total`, `population_growth`
- `gni_atlas_usd`, `gni_per_capita_atlas_usd`
- `life_expectancy`, `school_enrollment_secondary`

## Project Structure
```
├── models/
│   ├── linear_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── predictions_2023.csv       # Contains predictions and actual GDP data
├── data/
│   └── preprocessed_data.csv      # Preprocessed dataset
├── src/
│   └── data_preprocessing.py      # Preprocessing dataset
│   └── eda.py                     # EDA on dataset
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

## Model Evaluation
- **Root Mean Squared Error (RMSE)**: A metric used to evaluate the prediction error. Lower RMSE values indicate better performance.
- **Accuracy Percentage**: Measures the accuracy of the model as a percentage. A higher percentage indicates better performance.

### Sample Output
The output of the project displays a table of predicted GDP growth for 2023 alongside the actual GDP growth. It also provides a comparison of model performance using RMSE and accuracy percentage, as shown below:

| Model             | RMSE   | Accuracy (%) |
|-------------------|--------|--------------|
| Linear Regression | 3.8715 | 82.452       |
| Random Forest     | 2.6262 | 88.0966      |
| XGBoost           | 2.4839 | 88.7414      |

## Conclusion
- The project provides insights into how different machine learning models can predict GDP growth based on historical data.
- While the models show varying accuracy levels, there is potential for improvement through further optimization.

## Future Improvements
- **Hyperparameter Tuning**: Fine-tuning the hyperparameters of the models to enhance performance.
- **Feature Engineering**: Adding more relevant features such as inflation rate, unemployment rate, etc.
- **Advanced Models**: Exploring ensemble methods or deep learning techniques for better predictions.
- **Scalability**: Expanding the application to include additional economic indicators and broader temporal coverage.

---
Explore the potential of this project to analyze and predict GDP growth with interactive visualizations and robust machine learning models.
