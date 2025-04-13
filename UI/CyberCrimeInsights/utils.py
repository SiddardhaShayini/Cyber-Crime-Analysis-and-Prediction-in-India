import pandas as pd
import numpy as np
import joblib
import os
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

@st.cache_data
def load_data(file_path):
    """
    Load and cache the dataset
    """
    try:
        data = pd.read_csv(file_path)
        # Convert from wide to long format for easier analysis
        # Melt the dataframe to convert year columns to rows
        years_columns = [str(year) for year in range(2002, 2022)]
        years_columns = [col for col in years_columns if col in data.columns]
        
        # Check if we need to transform the data
        if 'Year' not in data.columns and len(years_columns) > 0:
            id_vars = ['State/UT']
            if 'Total' in data.columns:
                id_vars.extend(['Total', 'Total-Scale', 'State/UT_encoded', 'Mean_2002_2021', 'Std_2002_2021', 'YoY_2021'])
                
            melted_data = pd.melt(
                data, 
                id_vars=id_vars, 
                value_vars=years_columns,
                var_name='Year', 
                value_name='Cyber Crimes'
            )
            
            # Convert Year to integer
            melted_data['Year'] = melted_data['Year'].astype(int)
            
            # Sort by State/UT and Year
            melted_data = melted_data.sort_values(by=['State/UT', 'Year'])
            
            return melted_data
        else:
            return data
            
    except FileNotFoundError:
        st.error(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def load_models(arima_file, poly_file):
    """
    Load and cache the pre-trained models
    """
    try:
        # For demo purposes, we'll create simple models if the pickle files are placeholders
        with open(arima_file, 'r') as f:
            content = f.read()
            if content.startswith('dummy'):
                # Create dummy ARIMA model-like objects
                arima_models = {}
                return arima_models, create_poly_models()
                
        with open(poly_file, 'r') as f:
            content = f.read()
            if content.startswith('dummy'):
                # The polynomial models are already created by the arima check
                poly_models = create_poly_models()
                return {}, poly_models
                
        # If we're here, we have actual model files
        arima_models = joblib.load(arima_file)
        poly_models = joblib.load(poly_file)
        return arima_models, poly_models
        
    except FileNotFoundError as e:
        st.error(f"Error: Model file not found: {str(e)}")
        return {}, create_poly_models()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}, create_poly_models()

def create_poly_models():
    """
    Create polynomial regression models for each state based on the data
    """
    try:
        # Load the data
        data = pd.read_csv('processed_dataset.csv')
        
        # Extract years as columns
        year_columns = [str(year) for year in range(2002, 2022) if str(year) in data.columns]
        
        # Create models for each state
        poly_models = {}
        
        for _, row in data.iterrows():
            state = row['State/UT']
            if state in ['TOTAL (STATES)', 'TOTAL (UTS)', 'TOTAL (ALL INDIA)']:
                continue
                
            # Get crime values for each year
            X = np.array(range(2002, 2002 + len(year_columns))).reshape(-1, 1)
            y = np.array([row[year] for year in year_columns])
            
            # Create polynomial features
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            
            # Fit model
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Store the model
            poly_models[state] = {
                'model': model,
                'poly': poly,
                'last_value': y[-1],
                'last_year': 2021
            }
            
        return poly_models
    except Exception as e:
        st.warning(f"Error creating polynomial models: {str(e)}")
        return {}

def validate_year_input(year, min_year, max_year):
    """
    Validate that a year input is within the acceptable range
    """
    if year < min_year or year > max_year:
        return False, f"Year must be between {min_year} and {max_year}"
    return True, ""

def validate_future_year_input(year, min_future_year=2025):
    """
    Validate that a future year input is at least the minimum future year
    """
    if year < min_future_year:
        return False, f"Future year must be {min_future_year} or later"
    return True, ""

def forecast_state(state, future_year, arima_models, poly_models, max_historical_year):
    """
    Generate a forecast for a specific state and future year using a hybrid approach
    that combines both ARIMA and polynomial regression models
    """
    # Get polynomial prediction
    poly_pred = forecast_with_poly(state, future_year, poly_models)
    
    # Get ARIMA prediction if available
    arima_pred = 0
    if state in arima_models:
        try:
            model = arima_models[state]
            # Calculate steps from the last historical year
            steps = future_year - max_historical_year
            # Get the forecast
            arima_pred = model.get_forecast(steps=steps).predicted_mean.values[-1]
            arima_pred = max(0, arima_pred)
        except:
            # If ARIMA prediction fails, just use polynomial
            arima_pred = 0
    
    # Hybrid model: If both predictions are available, take their average
    # Otherwise, use whichever is available
    if arima_pred > 0 and poly_pred > 0:
        # Combine both models with weights (can be adjusted)
        hybrid_pred = 0.5 * arima_pred + 0.5 * poly_pred
    else:
        # Use whichever model provided a prediction
        hybrid_pred = max(arima_pred, poly_pred)
        
    return max(0, hybrid_pred)  # Ensure we don't predict negative values

def forecast_with_poly(state, future_year, poly_models):
    """
    Use polynomial model to forecast
    """
    if state not in poly_models:
        return 0
        
    model_data = poly_models[state]
    model = model_data['model']
    poly = model_data['poly']
    
    # Transform the input year
    X_future = poly.transform(np.array([[future_year]]))
    
    # Make prediction
    pred = model.predict(X_future)[0]
    
    # Ensure we don't predict negative values
    return max(0, pred)

def forecast_all_states(states, future_year, arima_models, poly_models, max_historical_year):
    """
    Generate forecasts for all states for a specific future year
    """
    predictions = {}
    for state in states:
        predictions[state] = forecast_state(state, future_year, arima_models, poly_models, max_historical_year)
    return predictions
