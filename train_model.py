``````import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Load the dataset
data_path = 'data/routes.csv'  # Replace with the path to your actual dataset
df = pd.read_csv(data_path)

# Check if the dataset is empty
if df.empty:
    print("The dataset is empty! Please check your CSV file.")
else:
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")

    # Encode categorical columns (weather_condition, origin_city, destination_city)
    label_encoder = LabelEncoder()
    df['weather_condition'] = label_encoder.fit_transform(df['weather_condition'])
    df['origin_city'] = label_encoder.fit_transform(df['origin_city'])
    df['destination_city'] = label_encoder.fit_transform(df['destination_city'])

    # Features (X) and target (y)
    X = df[['origin_city', 'destination_city', 'distance', 'fuel_consumption', 'weather_condition', 'safety_factor']]
    y = df['time_taken']

    # Check if there is enough data for splitting
    if len(df) > 1:
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the RandomForestRegressor model
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Fit the model to the training data
        model.fit(X_train, y_train)

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

        # Optionally, save the model predictions for training as well, if you want
        y_train_pred = model.predict(X_train)
        train_predictions_df = pd.DataFrame({
            'True Values': y_train,
            'Predicted Values': y_train_pred
        })
        train_predictions_df.to_csv('model/train_predictions.csv', index=False)

        # Save the trained model
        import joblib
        joblib.dump(model, 'model/route_model.pkl')

        print("Model training complete. Predictions saved to CSV.")
    else:
        print("Not enough data for training. Please check your dataset.")
