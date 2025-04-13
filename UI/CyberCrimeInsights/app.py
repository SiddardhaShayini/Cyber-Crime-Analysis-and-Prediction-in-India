import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
from utils import load_data, load_models, validate_year_input, validate_future_year_input, forecast_state, forecast_all_states

# Page configuration
st.set_page_config(
    page_title="Cyber Crime Forecasting and Analysis in India",
    page_icon="🔒",
    layout="wide"
)

# Load data and models
data = load_data('processed_dataset.csv')
arima_models, poly_models = load_models('arima_models.pkl', 'poly_models.pkl')

# Get list of states
states = sorted(data['State/UT'].unique())
years = sorted(data['Year'].unique())
min_year = min(years)
max_year = max(years)

# Sidebar for feature selection
st.sidebar.title("Features")
feature = st.sidebar.radio(
    "Select Analysis Feature",
    [
        "Home",
        "Year-wise Trend for a Particular State",
        "Analyze Cyber Crime for a Particular Year (All States)",
        "Forecast from Start Year to End Year for a Particular State",
        "Forecast for a Specific State & Specific Year",
        "Forecast for All States in a Future Year",
        "Forecast Top 5 States with Highest Expected Crimes in a Future Year",
        "Cyber Crime Forecast for India up to future year"
    ]
)

# Home page
if feature == "Home":
    st.title("Cyber Crime Forecasting and Analysis in India")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("About the Project")
        st.write("""
        This dashboard provides analytical tools to understand and predict cyber crime trends across India.
        Using historical data from 2002 to 2021, the system provides various forecasting capabilities
        to help policymakers, law enforcement agencies, and researchers understand future trends.
        
        The analyses include state-specific trends, yearly comparisons, and advanced forecasting
        using time series modeling techniques. All visualizations are interactive, allowing users
        to explore the data in detail.
        """)
        
        st.subheader("How to Use")
        st.write("""
        1. Select an analysis feature from the sidebar
        2. Enter required parameters such as state, year, or forecast period
        3. Explore the interactive charts and predictions
        4. Hover over data points to see detailed information
        """)
    
    with col2:
        st.subheader("Available Features")
        st.write("- Year-wise Trend Analysis")
        st.write("- State-wise Cyber Crime Analysis")
        st.write("- Multi-year Forecasting")
        st.write("- State-specific Future Predictions")
        st.write("- All-India Forecasting")
        st.write("- Top Cyber Crime Hotspot Prediction")
        
    # Overview chart - Total cyber crimes by year for all India
    st.subheader("Overview: Total Cyber Crimes in India (2002-2021)")
    
    # Group by year and sum the cyber crimes
    yearly_total = data.groupby('Year')['Cyber Crimes'].sum().reset_index()
    
    fig = px.line(
        yearly_total, 
        x='Year', 
        y='Cyber Crimes',
        markers=True,
        title="Total Cyber Crimes in India (2002-2021)"
    )
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Cyber Crimes",
        xaxis=dict(tickmode='linear'),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# Feature 1: Year-wise Trend for a Particular State
elif feature == "Year-wise Trend for a Particular State":
    st.title("Year-wise Trend for a Particular State")
    
    selected_state = st.selectbox("Select State/UT", states)
    
    if selected_state:
        state_data = data[data['State/UT'] == selected_state]
        
        fig = px.line(
            state_data, 
            x='Year', 
            y='Cyber Crimes',
            markers=True,
            title=f"Cyber Crime Trend in {selected_state} (2002-2021)"
        )
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Number of Cyber Crimes",
            xaxis=dict(tickmode='linear'),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        st.subheader("Yearly Data")
        st.dataframe(state_data[['Year', 'Cyber Crimes']].set_index('Year'))

# Feature 2: Analyze Cyber Crime for a Particular Year (All States)
elif feature == "Analyze Cyber Crime for a Particular Year (All States)":
    st.title("Analyze Cyber Crime for a Particular Year (All States)")
    
    selected_year = st.slider("Select Year", min_value=min_year, max_value=max_year, value=max_year)
    
    if selected_year:
        year_data = data[data['Year'] == selected_year].sort_values(by='Cyber Crimes', ascending=False)
        
        # Bar chart
        fig = px.bar(
            year_data, 
            x='State/UT', 
            y='Cyber Crimes',
            title=f"Cyber Crimes across Indian States/UTs in {selected_year}",
            color='Cyber Crimes',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            xaxis_title="State/UT",
            yaxis_title="Number of Cyber Crimes",
            xaxis={'categoryorder':'total descending'},
            hovermode="closest"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Choropleth map of India (if wanted in future)
        
        # Top 5 and Bottom 5 states
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Top 5 States with Highest Cyber Crimes in {selected_year}")
            st.dataframe(year_data.head(5)[['State/UT', 'Cyber Crimes']])
        
        with col2:
            st.subheader(f"Bottom 5 States with Lowest Cyber Crimes in {selected_year}")
            st.dataframe(year_data.tail(5)[['State/UT', 'Cyber Crimes']])

# Feature 3: Forecast from Start Year to End Year for a Particular State
elif feature == "Forecast from Start Year to End Year for a Particular State":
    st.title("Forecast from Start Year to End Year for a Particular State")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_state = st.selectbox("Select State/UT", states)
    
    with col2:
        start_year = st.number_input("Start Year (≥2025)", min_value=2025, max_value=2050, value=2025)
    
    with col3:
        end_year = st.number_input("End Year (> Start Year)", min_value=start_year+1, max_value=2050, value=2030)
    
    if selected_state and start_year < end_year:
        # Get historical data
        state_data = data[data['State/UT'] == selected_state]
        
        # Generate future years
        future_years = list(range(2022, end_year + 1))
        
        # Make predictions using the hybrid approach (combine both models)
        predictions = []
        for year in future_years:
            # Use forecast_state which now implements the hybrid approach
            pred = forecast_state(selected_state, year, arima_models, poly_models, max_year)
            predictions.append(max(0, pred))
        
        # Create a dataframe with historical and predicted data
        historical_df = state_data[['Year', 'Cyber Crimes']]
        forecast_df = pd.DataFrame({
            'Year': future_years,
            'Cyber Crimes': predictions
        })
        
        # Filter forecast_df to only include years from start_year to end_year
        forecast_df = forecast_df[(forecast_df['Year'] >= start_year) & (forecast_df['Year'] <= end_year)]
        
        # Plot
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_df['Year'],
            y=historical_df['Cyber Crimes'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Add forecast data
        fig.add_trace(go.Scatter(
            x=forecast_df['Year'],
            y=forecast_df['Cyber Crimes'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dot')
        ))
        
        fig.update_layout(
            title=f"Cyber Crime Forecast for {selected_state} ({start_year} to {end_year})",
            xaxis_title="Year",
            yaxis_title="Number of Cyber Crimes",
            xaxis=dict(tickmode='linear'),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show forecast data
        st.subheader("Forecast Data")
        st.dataframe(forecast_df.set_index('Year'))

# Feature 4: Forecast for a Specific State & Specific Year
elif feature == "Forecast for a Specific State & Specific Year":
    st.title("Forecast for a Specific State & Specific Year")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_state = st.selectbox("Select State/UT", states)
    
    with col2:
        future_year = st.number_input("Future Year (≥2025)", min_value=2025, max_value=2050, value=2025)
    
    if selected_state and future_year >= 2025:
        # Get historical data
        state_data = data[data['State/UT'] == selected_state]
        
        # Make prediction
        prediction = forecast_state(selected_state, future_year, arima_models, poly_models, max_year)
        
        # Display the prediction
        st.subheader(f"Prediction for {selected_state} in {future_year}")
        
        # Create a metric display
        st.metric(
            label=f"Predicted Cyber Crimes in {future_year}",
            value=f"{int(prediction)}",
            delta=f"{int(prediction - state_data['Cyber Crimes'].iloc[-1])} from {max_year}"
        )
        
        # Get historical data for visualization
        historical_df = state_data[['Year', 'Cyber Crimes']]
        
        # Create a dataframe with one row for the prediction
        forecast_df = pd.DataFrame({
            'Year': [future_year],
            'Cyber Crimes': [prediction]
        })
        
        # Plot
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_df['Year'],
            y=historical_df['Cyber Crimes'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Add forecast point
        fig.add_trace(go.Scatter(
            x=forecast_df['Year'],
            y=forecast_df['Cyber Crimes'],
            mode='markers',
            name='Forecast',
            marker=dict(color='red', size=12, symbol='diamond')
        ))
        
        # Add a dotted line connecting the last historical point to the forecast
        fig.add_trace(go.Scatter(
            x=[historical_df['Year'].iloc[-1], future_year],
            y=[historical_df['Cyber Crimes'].iloc[-1], prediction],
            mode='lines',
            line=dict(color='red', dash='dot'),
            showlegend=False
        ))
        
        fig.update_layout(
            title=f"Cyber Crime Forecast for {selected_state} in {future_year}",
            xaxis_title="Year",
            yaxis_title="Number of Cyber Crimes",
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Feature 5: Forecast for All States in a Future Year
elif feature == "Forecast for All States in a Future Year":
    st.title("Forecast for All States in a Future Year")
    
    future_year = st.number_input("Future Year (≥2025)", min_value=2025, max_value=2050, value=2025)
    
    if future_year >= 2025:
        # Make predictions for all states
        predictions = forecast_all_states(states, future_year, arima_models, poly_models, max_year)
        
        # Create a dataframe with the predictions
        forecast_df = pd.DataFrame({
            'State/UT': states,
            'Predicted Cyber Crimes': [predictions[state] for state in states]
        }).sort_values(by='Predicted Cyber Crimes', ascending=False)
        
        # Bar chart
        fig = px.bar(
            forecast_df, 
            x='State/UT', 
            y='Predicted Cyber Crimes',
            title=f"Predicted Cyber Crimes across Indian States/UTs in {future_year}",
            color='Predicted Cyber Crimes',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            xaxis_title="State/UT",
            yaxis_title="Predicted Number of Cyber Crimes",
            xaxis={'categoryorder':'total descending'},
            hovermode="closest"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the data
        st.subheader(f"Predicted Cyber Crimes for All States in {future_year}")
        st.dataframe(forecast_df)

# Feature 6: Forecast Top 5 States with Highest Expected Crimes in a Future Year
elif feature == "Forecast Top 5 States with Highest Expected Crimes in a Future Year":
    st.title("Forecast Top 5 States with Highest Expected Crimes in a Future Year")
    
    future_year = st.number_input("Future Year (≥2025)", min_value=2025, max_value=2050, value=2025)
    
    if future_year >= 2025:
        # Make predictions for all states
        predictions = forecast_all_states(states, future_year, arima_models, poly_models, max_year)
        
        # Create a dataframe with the predictions
        forecast_df = pd.DataFrame({
            'State/UT': states,
            'Predicted Cyber Crimes': [predictions[state] for state in states]
        }).sort_values(by='Predicted Cyber Crimes', ascending=False)
        
        # Get top 5 states
        top_5_states = forecast_df.head(5)
        
        # Bar chart for top 5
        fig = px.bar(
            top_5_states, 
            x='State/UT', 
            y='Predicted Cyber Crimes',
            title=f"Top 5 States with Highest Predicted Cyber Crimes in {future_year}",
            color='Predicted Cyber Crimes',
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            xaxis_title="State/UT",
            yaxis_title="Predicted Number of Cyber Crimes",
            xaxis={'categoryorder':'total descending'},
            hovermode="closest"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the data for top 5
        st.subheader(f"Top 5 States with Highest Predicted Cyber Crimes in {future_year}")
        st.dataframe(top_5_states)
        
        # Let's also create a pie chart for the distribution
        fig2 = px.pie(
            top_5_states, 
            values='Predicted Cyber Crimes', 
            names='State/UT',
            title=f"Distribution of Cyber Crimes Among Top 5 States in {future_year}"
        )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig2, use_container_width=True)

# Feature 7: Cyber Crime Forecast for India up to future year
elif feature == "Cyber Crime Forecast for India up to future year":
    st.title("Cyber Crime Forecast for India up to future year")
    
    future_year = st.number_input("Future Year (≥2025)", min_value=2025, max_value=2050, value=2030)
    
    if future_year >= 2025:
        # Get historical all-India data by summing across states for each year
        yearly_total = data.groupby('Year')['Cyber Crimes'].sum().reset_index()
        
        # Generate future years
        future_years = list(range(2022, future_year + 1))
        
        # For each future year, forecast for all states and sum
        future_totals = []
        for year in future_years:
            predictions = forecast_all_states(states, year, arima_models, poly_models, max_year)
            total = sum(predictions.values())
            future_totals.append(total)
        
        # Create a dataframe for the forecast
        forecast_df = pd.DataFrame({
            'Year': future_years,
            'Predicted Cyber Crimes': future_totals
        })
        
        # Plot
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=yearly_total['Year'],
            y=yearly_total['Cyber Crimes'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Add forecast data
        fig.add_trace(go.Scatter(
            x=forecast_df['Year'],
            y=forecast_df['Predicted Cyber Crimes'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dot')
        ))
        
        fig.update_layout(
            title=f"All-India Cyber Crime Forecast up to {future_year}",
            xaxis_title="Year",
            yaxis_title="Number of Cyber Crimes",
            xaxis=dict(tickmode='linear'),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate growth statistics
        last_historical = yearly_total['Cyber Crimes'].iloc[-1]
        last_forecast = forecast_df['Predicted Cyber Crimes'].iloc[-1]
        growth_absolute = last_forecast - last_historical
        growth_percent = (growth_absolute / last_historical) * 100
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label=f"Cyber Crimes in {max_year} (Last Historical)",
                value=f"{int(last_historical)}"
            )
        
        with col2:
            st.metric(
                label=f"Predicted Cyber Crimes in {future_year}",
                value=f"{int(last_forecast)}"
            )
        
        with col3:
            st.metric(
                label=f"Growth from {max_year} to {future_year}",
                value=f"{growth_percent:.2f}%",
                delta=f"{int(growth_absolute)} crimes"
            )
        
        # Show forecast data
        st.subheader("Forecast Data")
        st.dataframe(forecast_df.set_index('Year'))
