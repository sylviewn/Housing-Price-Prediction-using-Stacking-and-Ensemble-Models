Housing Price Prediction using Stacking and Ensemble Models

This repository contains code for predicting housing prices using various machine learning models, including a Stacking Regressor and other ensemble techniques. The project utilizes multiple algorithms such as Random Forest, XGBoost, LightGBM, and Gradient Boosting, along with fine-tuned hyperparameters for the best performance.

Project Overview

The goal of this project is to predict housing prices based on a dataset of housing features. The project demonstrates the power of ensemble learning by combining multiple models to create a more robust and accurate Stacking Regressor.

Models Used

    Random Forest (Tuned and Non-Tuned)
    XGBoost
    LightGBM
    Gradient Boosting (Tuned and Non-Tuned)
    Stacking Regressor (Final combined model)

The Stacking Regressor is shown to provide the best predictive performance by combining the strengths of these individual models.

Data

The dataset used for this project contains features like:

    Median income
    Location features (longitude, latitude)
    Housing characteristics (total rooms, total bedrooms, population, households, etc.)

The target variable is the median house value.

Feature Engineering

Various feature engineering steps have been applied, including:

    Scaling numeric features
    Interaction terms
    Polynomial features
    Feature selection using Random Forest Importance

Results

The Stacking Regressor outperformed individual models such as Random Forest, XGBoost, and LightGBM. The key performance metrics include:

    Mean Absolute Error (MAE)
    Mean Squared Error (MSE)
    R-squared (R²)

Best Model Performance:

    MAE: 35,266
    MSE: 2,629,630,993 
    R²:  0.7449

Requirements

The code relies on several Python libraries for machine learning and data processing. You can install the required dependencies using the requirements.txt file:

    pip install -r requirements.txt

Key libraries:

    scikit-learn
    xgboost
    lightgbm
    joblib
    matplotlib
    seaborn
    pandas
    numpy

How to Run

Clone the repository:

    git clone https://github.com/your-username/housing-price-prediction.git
    cd housing-price-prediction

Install the required libraries:

    pip install -r requirements.txt

    3. Prepare your dataset (if not included):
        - Ensure that your dataset is in the correct format and is preprocessed as needed.

    4. Run the notebook or Python scripts to train models and make predictions:
        - Open the Jupyter notebook housing_price_prediction.ipynb to explore the models and results.
        - Alternatively, you can run the scripts directly to train models and generate predictions.

Saving and Loading the Model

Save the Model

After training the Stacking Regressor, you can save it for future use to avoid retraining every time:

     import joblib

# Save the best stacking model
     joblib.dump(best_stacking_model, 'best_stacking_model.pkl')

This will save the trained model to a file named best_stacking_model.pkl.

Load the Model

To load the saved model and use it for making predictions:


# Load the saved stacking model
     loaded_model = joblib.load('best_stacking_model.pkl')

# Use the loaded model to make predictions
     y_pred_loaded = loaded_model.predict(X_test)

Project Structure

    - housing_price_prediction.ipynb: Main Jupyter notebook with the code for model training, tuning, and evaluation.
    - engineered_features.csv: Preprocessed dataset with selected and engineered features.
    - requirements.txt: List of libraries required to run the project.
    - stacking_model_housing.pkl: Saved Stacking model (after training) for use in future predictions.

Future Work

    - Explore more advanced feature engineering techniques.
    - Tune hyperparameters further using Bayesian optimization or RandomizedSearchCV.
    - Extend the model to predict housing prices for other datasets.

License

This project is licensed under the MIT License. See the LICENSE file for more details.-
