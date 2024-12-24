import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Hardcoded login credentials
USERNAME = "DSS"
PASSWORD = "12345"

# Load the dataset for training the model
data_path = 'data/routes.csv'  # Replace with your dataset's path
df = pd.read_csv(data_path)

# Encode categorical columns
label_encoder = LabelEncoder()
df['weather_condition'] = label_encoder.fit_transform(df['weather_condition'])
df['origin_city'] = label_encoder.fit_transform(df['origin_city'])
df['destination_city'] = label_encoder.fit_transform(df['destination_city'])

# Features (X) and target (y)
X = df[['origin_city', 'destination_city', 'distance', 'fuel_consumption', 'weather_condition', 'safety_factor']]
y = df['time_taken']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model to a .pkl file
joblib.dump(model, 'model/route_model.pkl')

# Save the model's feature importances to a CSV file
importances = model.feature_importances_
features = X.columns
importances_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
})
importances_df.to_csv('model/feature_importances.csv', index=False)

# Save the model's predictions on the test set to a CSV file
y_pred = model.predict(X_test)
predictions_df = pd.DataFrame({
    'True Values': y_test,
    'Predicted Values': y_pred
})
predictions_df.to_csv('model/predictions.csv', index=False)

# Optionally, save the model predictions for training as well
y_train_pred = model.predict(X_train)
train_predictions_df = pd.DataFrame({
    'True Values': y_train,
    'Predicted Values': y_train_pred
})
train_predictions_df.to_csv('model/train_predictions.csv', index=False)

# Load the trained model for Flask predictions
model = joblib.load('model/route_model.pkl')

# Example city data with latitude and longitude
cities = {
    "Chennai": (13.0827, 80.2707),
    "Port Blair": (11.6667, 92.7500),
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    # Validate user credentials
    if username == USERNAME and password == PASSWORD:
        return redirect(url_for('route_form'))
    else:
        return "Invalid credentials. Please try again."

@app.route('/route_form')
def route_form():
    return render_template('route_form.html')

@app.route('/predict_route', methods=['POST'])
def predict_route():
    origin = request.form['origin']
    destination = request.form['destination']
    distance = float(request.form['distance'])
    fuel_consumption = float(request.form['fuel_consumption'])
    weather_condition = request.form['weather_condition']
    safety_factor = float(request.form['safety_factor'])
    
    # Check if the origin and destination are valid cities
    if origin not in cities or destination not in cities:
        return "Invalid city names. Please try again."
    
    # Encode the input data
    origin_encoded = 0 if origin == "Chennai" else 1
    destination_encoded = 0 if destination == "Chennai" else 1
    weather_encoded = 0 if weather_condition == "Clear" else 1

    # Prepare the input data for prediction
    input_data = pd.DataFrame([{
        'origin_city': origin_encoded,
        'destination_city': destination_encoded,
        'distance': distance,
        'fuel_consumption': fuel_consumption,
        'weather_condition': weather_encoded,
        'safety_factor': safety_factor
    }])

    # Get the prediction from the model
    prediction = model.predict(input_data)[0]

    # Get latitude and longitude of the origin and destination cities
    origin_lat, origin_lon = cities[origin]
    dest_lat, dest_lon = cities[destination]
    
    # Create a folium map centered between the origin and destination
    map_route = folium.Map(location=[(origin_lat + dest_lat) / 2, (origin_lon + dest_lon) / 2], zoom_start=6)
    
    # Add markers for the origin and destination
    folium.Marker([origin_lat, origin_lon], popup=f'Origin: {origin}').add_to(map_route)
    folium.Marker([dest_lat, dest_lon], popup=f'Destination: {destination}').add_to(map_route)
    
    # Add a line for the route (Optional)
    folium.PolyLine([(origin_lat, origin_lon), (dest_lat, dest_lon)], color="blue", weight=2.5, opacity=1).add_to(map_route)
    
    # Save the map as an HTML file
    map_route.save('templates/route_map.html')
    
    return render_template('route_map.html', time_taken=prediction)

if __name__ == '__main__':
    app.run(debug=True)
