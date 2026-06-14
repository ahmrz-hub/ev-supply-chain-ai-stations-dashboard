# src/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train_and_export_knn():
    print("Loading dataset and isolating features for KNN...")
    
    df = pd.read_csv("data/cleaned_master_data.csv")
    
    # Define features (X) and target variable (y)
    features = ['year', 'ev_stock', 'ev_sales', 'prev_year_stations', 'prev_year_ev_stock', 'battery_demand_mwh_est']
    target = 'total_stations'
    
    X = df[features]
    y = df[target]
    
    # Split data (80% Train, 20% Testing Validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # CRITICAL STEP FOR KNN: Scaling Features
    # KNN relies on physical distance between data points. Because EV Stock is in the millions
    # and Year is around 2026, we MUST scale them so one feature doesn't dominate the distance calculation.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and Train KNN Regressor (using K=5 neighbors as a baseline)
    print("🏋️ Training K-Nearest Neighbors Regressor (KNN)...")
    model = KNeighborsRegressor(n_neighbors=5, weights='distance')
    model.fit(X_train_scaled, y_train)
    
    # Run Evaluation
    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n📊 --- KNN MODEL METRICS ---")
    print(f"   Mean Absolute Error (MAE): {mae:.2f} stations")
    print(f"   R² Accuracy Score: {r2 * 100:.2f}%")
    print("────────────────────────\n")
    
    # Save both the model AND the scaler artifact
    # The dashboard will need the scaler to process the user inputs exactly like the training data
    joblib.dump(model, "src/infrastructure_model.joblib")
    joblib.dump(scaler, "src/scaler.pkl")
    print("Complete! KNN Model and Scaler files serialized successfully.")

if __name__ == "__main__":
    train_and_export_knn()
