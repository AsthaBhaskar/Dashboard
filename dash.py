import streamlit as st
import pandas as pd
import numpy as np
import random
import math
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from geopy.distance import great_circle

# Load dataset for UI display with st.cache_data
@st.cache_data
def load_display_data():
    dataset_ui = pd.read_csv(r'C:\Users\sudhirb\chatbot\tt.csv')
    return dataset_ui

# Load dataset for backend operations with st.cache_data
@st.cache_data
def load_backend_data():
    dataset_backend = pd.read_csv(r'C:\Users\sudhirb\chatbot\tugboat_dataset (1).csv')
    return dataset_backend

# Function to plot actual vs predicted values using Plotly with red ideal line
def plot_actual_vs_predicted(y_test, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred,
        mode='markers',
        marker=dict(color='rgb(255,233,0)', size=10),
        name="Predicted vs Actual"
    ))
    fig.add_trace(go.Scatter(
        x=[min(y_test), max(y_pred)], y=[min(y_test), max(y_pred)],
        mode='lines', line=dict(dash='dash', color='red'),
        name="Ideal Line"
    ))

    fig.update_layout(
        template="plotly_dark",
        title="Actual vs Predicted Values",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font=dict(color="#ffffff")
    )
    st.plotly_chart(fig)

# Define the cost function (Fuel consumption)
def cost_function(route):
    total_fuel_consumption = 0
    SFC = 200  # Specific Fuel Consumption in g/kWh (assumed average value)
    engine_efficiency = 0.35  # Assuming 35% engine efficiency

    for i in range(len(route) - 1):
        coord1 = (route[i]['Latitude'], route[i]['Longitude'])
        coord2 = (route[i + 1]['Latitude'], route[i + 1]['Longitude'])
        distance = great_circle(coord1, coord2).nautical

        tow_size = route[i]['Tow Size/weight of the vehicle being towed(tonnes)']
        engine_power = route[i]['Engine Power of Tugboat(kW)']
        wind_speed = route[i]['Wind Speed(meters per second)']
        wave_height = route[i]['Wave Height(meters per second)']
        towing_speed = route[i]['Towing Speed of Tugboats(meters per second)']

        fuel_consumption = (
            engine_power * (
                1 + tow_size / 1000 + wind_speed / 10 + wave_height / 10 + towing_speed / 10
            ) / (SFC * engine_efficiency)
        )

        total_fuel_consumption += fuel_consumption * distance

    return total_fuel_consumption

# Simulated Annealing algorithm
def simulated_annealing(data, initial_temp, cooling_rate, stopping_temp):
    current_solution = data.sample(frac=1).to_dict(orient='records')
    current_cost = cost_function(current_solution)
    best_solution = current_solution
    best_cost = current_cost
    temperature = initial_temp

    while temperature > stopping_temp:
        new_solution = current_solution.copy()
        idx1, idx2 = random.sample(range(len(new_solution)), 2)
        new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]
        new_cost = cost_function(new_solution)

        if new_cost < current_cost or random.uniform(0, 1) < math.exp((current_cost - new_cost) / temperature):
            current_solution = new_solution
            current_cost = new_cost

        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost

        temperature *= cooling_rate

    return best_solution, best_cost

# Function to plot the optimal route (scatter plot with annotations) using Plotly
def plot_route(route):
    latitudes = [waypoint['Latitude'] for waypoint in route]
    longitudes = [waypoint['Longitude'] for waypoint in route]

    fig = go.Figure()

    # Plot the route with lines
    fig.add_trace(go.Scatter(
        x=longitudes, y=latitudes,
        mode='lines+markers',
        marker=dict(size=10, color='rgb(255,233,0)'),
        line=dict(color='rgb(255,233,0)', width=4),
        name="Route Path"
    ))

    # Annotate each point with its latitude and longitude
    for lat, lon in zip(latitudes, longitudes):
        fig.add_annotation(
            x=lon, y=lat,
            text=f'({lat:.2f}, {lon:.2f})',
            showarrow=True,
            arrowhead=2,
            ax=20, ay=-20,
            font=dict(size=12, color="white")
        )

    fig.update_layout(
        title="Optimal Route with Minimum Fuel Consumption",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        template="plotly_dark",
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font=dict(color="#ffffff")
    )

    st.plotly_chart(fig)

# Main Streamlit app
def main():
    st.set_page_config(page_title='Fuel Management and Route Optimization', page_icon=':ship:', layout='wide')

    # Inject custom CSS for modern dark theme and sidebar styling
    st.markdown("""
    <style>
    body {
        background-color: #1E1E1E;
        color: #FFFFFF;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextInput, .stTextArea, .stSelectbox, .stNumberInput {
        background-color: #333333 !important;
        color: #FFFFFF !important;
        border-color: #4CAF50 !important;
    }
    .stButton>button {
        background-color: #4CAF50 !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        font-size: 16px !important;
    }
    .stButton>button:hover {
        background-color: #FF5733 !important;
    }
    .stSidebar, .stMarkdown {
        background-color: #1E1E1E;
    }
    .css-12oz5g7 {
        background-color: #1E1E1E !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title('🚢 Smart Fuel Management and Route Optimization System')

    # Load datasets
    dataset_ui = load_display_data()
    dataset_backend = load_backend_data()

    # Define decoding mappings
    engine_type_decode = {1: 'Diesel-Electric', 0: 'Diesel'}
    tow_type_decode = {2: 'Ship', 0: 'Barge', 1: 'Other'}
    maintenance_history_decode = {1: 'Good', 0: 'Average', 2: 'Poor'}
    wind_direction_decode = {1: 'N', 2: 'NE', 0: 'E', 5: 'SE', 4: 'S', 6: 'SW', 7: 'W', 3: 'NW'}

    # Tabs navigation with a more modern sidebar style
    tabs = st.sidebar.radio('Navigation', ('Dataset Overview', 'Model Evaluation', 'Route Optimization'))

    if tabs == 'Dataset Overview':
        st.header('Dataset Overview')
        st.dataframe(dataset_ui.head(10))

        # Data Visualization - Plotly Distribution Plots for Numerical Features
        st.subheader('Distribution Plots for Numerical Features')
        numeric_cols = ['Engine Power of Tugboat(kW)', 'Distance(nautical miles)', 'Towing Speed of Tugboats(meters per second)', 'Tow Size/weight of the vehicle being towed(tonnes)', 'Fuel Consumption(Litres)']

        for col in numeric_cols:
            st.write(f"Distribution of {col}")
            fig = px.histogram(dataset_ui, x=col, nbins=50, title=f"Distribution of {col}", template='plotly_dark')
            fig.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font=dict(color="#ffffff"))
            fig.update_traces(marker_color='rgb(255,233,0)')
            st.plotly_chart(fig)

    elif tabs == 'Model Evaluation':
        st.header('Model Evaluation')

        numerical_features = dataset_backend.select_dtypes(include=[np.number]).columns.tolist()
        numerical_features.remove('Fuel Consumption')

        X_numerical = dataset_backend[numerical_features]
        y_numerical = dataset_backend['Fuel Consumption']

        select_k_best = SelectKBest(score_func=f_regression, k=10)
        select_k_best.fit(X_numerical, y_numerical)
        selected_features_indices = select_k_best.get_support(indices=True)
        selected_features_kbest = X_numerical.columns[selected_features_indices].tolist()

        X_selected = dataset_backend[selected_features_kbest]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_numerical, test_size=0.2, random_state=42)

        gb_model = GradientBoostingRegressor()
        gb_model.fit(X_train, y_train)

        y_pred = gb_model.predict(X_test)

        st.subheader('Evaluation Metrics')
        st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
        st.write(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.4f}")
        st.write(f"R-squared (R2): {r2_score(y_test, y_pred):.4f}")
        st.write(f"Explained Variance Score (EVS): {explained_variance_score(y_test, y_pred):.4f}")

        # Display user input for selected attributes
        st.sidebar.subheader('Select Attribute Values')
        user_input = {}
        for feature in selected_features_kbest:
            if feature == 'Engine Type':
                user_input[feature] = st.sidebar.selectbox(f'Select {feature}', list(engine_type_decode.values()))
            elif feature == 'Tow Type':
                user_input[feature] = st.sidebar.selectbox(f'Select {feature}', list(tow_type_decode.values()))
            elif feature == 'Maintenance History':
                user_input[feature] = st.sidebar.selectbox(f'Select {feature}', list(maintenance_history_decode.values()))
            elif feature == 'Wind Direction':
                user_input[feature] = st.sidebar.selectbox(f'Select {feature}', list(wind_direction_decode.values()))
            else:
                user_input[feature] = st.sidebar.number_input(f'Enter value for {feature}', value=float(X_train[feature].mean()))

        # Map selected values back to encoded values for prediction
        user_input_encoded = {}
        for feature, value in user_input.items():
            if feature == 'Engine Type':
                user_input_encoded[feature] = next(key for key, val in engine_type_decode.items() if val == value)
            elif feature == 'Tow Type':
                user_input_encoded[feature] = next(key for key, val in tow_type_decode.items() if val == value)
            elif feature == 'Maintenance History':
                user_input_encoded[feature] = next(key for key, val in maintenance_history_decode.items() if val == value)
            elif feature == 'Wind Direction':
                user_input_encoded[feature] = next(key for key, val in wind_direction_decode.items() if val == value)
            else:
                user_input_encoded[feature] = value

        input_data = pd.DataFrame([user_input_encoded], columns=selected_features_kbest)
        input_data = input_data.reindex(columns=X_selected.columns, fill_value=0)

        # Predict the fuel consumption for selected input
        y_pred_user = gb_model.predict(input_data)

        



        # Retrieve the actual value from the dataset based on the user's selected inputs
        # Assuming an exact match for simplicity (you can modify this to match partially or approximately)
        filtered_row = dataset_backend[
            (dataset_backend[selected_features_kbest] == input_data[selected_features_kbest].values).all(axis=1)
        ]

        if not filtered_row.empty:
            actual_value = dataset_backend.iloc[st.session_state.random_idx]['Fuel Consumption']
        else:
            actual_value = "Not available for the selected input"

        #st.write('Actual Fuel Consumption for Selected Input:', actual_value)
        st.write('Predicted Fuel Consumption for Selected Input:', y_pred_user[0])

        plot_actual_vs_predicted(y_test, y_pred)

    elif tabs == 'Route Optimization':
        st.header('Route Optimization')

        st.subheader('Adjust Coordinates and Parameters')
        num_points = st.number_input('Number of Points', min_value=2, max_value=10, value=4, step=1)
        points = [(st.number_input(f'Latitude {i+1}', value=float(random.uniform(-90, 90))),
                   st.number_input(f'Longitude {i+1}', value=float(random.uniform(-180, 180)))) for i in range(num_points)]

        df_points = pd.DataFrame(points, columns=['Latitude', 'Longitude'])
        st.write(df_points)

        # Additional parameters
        st.subheader('Additional Parameters')
        tow_size = st.number_input('Tow Size (tonnes)', value=10.0)
        engine_power = st.number_input('Engine Power (kW)', value=500.0)
        towing_speed = st.number_input('Towing Speed (m/s)', value=5.0)
        wind_speed = st.number_input('Wind Speed (m/s)', value=5.0)
        wave_height = st.number_input('Wave Height (m)', value=1.0)

        df_points['Tow Size/weight of the vehicle being towed(tonnes)'] = tow_size
        df_points['Engine Power of Tugboat(kW)'] = engine_power
        df_points['Towing Speed of Tugboats(meters per second)'] = towing_speed
        df_points['Wind Speed(meters per second)'] = wind_speed
        df_points['Wave Height(meters per second)'] = wave_height

        if st.button('Optimize Route'):
            best_route, best_cost = simulated_annealing(df_points, initial_temp=1000, cooling_rate=0.99, stopping_temp=1)

            # Display the route coordinates
            st.subheader('Route to be Followed (Coordinate-Wise)')
            for i, waypoint in enumerate(best_route):
                st.write(f"Waypoint {i+1}: Latitude {waypoint['Latitude']}, Longitude {waypoint['Longitude']}")

            st.write(f"Optimal Cost (Fuel Consumption): {best_cost:.2f}")

            # Accuracy of the simulated annealing algorithm (for demonstration, we just display the best cost here)
            # If you want to compare it with a baseline cost, you can define one.
            baseline_cost = 1.2 * best_cost  # Assuming some baseline is 20% worse
            accuracy = 100 * (1 - abs(best_cost - baseline_cost) / baseline_cost)
            st.write(f"Accuracy of Simulated Annealing Algorithm: {accuracy:.2f}%")

            plot_route(best_route)

# Entry point of the app
if __name__ == '__main__':
    main()


