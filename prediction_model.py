# Step 3: Model Training - Simple Version
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class SimpleStockPredictor:
    def __init__(self):
        self.data = None
        self.models = {}
        self.scaler = MinMaxScaler()
        self.predictions = {}
        
    def prepare_data(self, symbol="AAPL", period="5y"):  
        """Get and prepare data for modeling"""
        print(" PREPARING DATA FOR MODELING")
        print("=" * 40)
        
        # Get data
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        data.reset_index(inplace=True)
        
        # Add basic features
        data['MA_5'] = data['Close'].rolling(5).mean()
        data['MA_10'] = data['Close'].rolling(10).mean()
        data['Price_Change'] = data['Close'].pct_change()
        data['Volume_MA'] = data['Volume'].rolling(10).mean()
        data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close']
        
        # Add simple lag features
        data['Close_lag_1'] = data['Close'].shift(1)
        data['Close_lag_2'] = data['Close'].shift(2)
        
        # Remove rows with NaN
        data.dropna(inplace=True)
        
        print(f"✅ Prepared {len(data)} rows of data")
        print(f" Features available: {list(data.columns)}")
        
        self.data = data
        return data
    
    def create_features_and_target(self):
        """Create feature matrix (X) and target variable (y)"""
        print("\n CREATING FEATURES AND TARGET")
        print("=" * 40)
        
        # Features to use for prediction (exclude Date and target)
        feature_columns = ['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_10', 
                          'Price_Change', 'Volume_MA', 'High_Low_Pct', 
                          'Close_lag_1', 'Close_lag_2']
        
        X = self.data[feature_columns]
        y = self.data['Close']  # This is what we want to predict
        
        print(f" Feature matrix shape: {X.shape}")
        print(f" Target vector shape: {y.shape}")
        print(f" Features used: {feature_columns}")
        
        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False  
        )
        
        print(f" Training set: {X_train.shape[0]} samples")
        print(f" Test set: {X_test.shape[0]} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_linear_model(self, X_train, y_train):
        """Train a simple Linear Regression model"""
        print("\n TRAINING LINEAR REGRESSION MODEL")
        print("=" * 40)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        self.models['Linear Regression'] = model
        print("✅ Linear Regression model trained!")
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train a Random Forest model"""
        print("\n TRAINING RANDOM FOREST MODEL")
        print("=" * 40)
        
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        self.models['Random Forest'] = model
        print("✅ Random Forest model trained!")
        return model
    
    def evaluate_models(self, X_test, y_test):
        """Test how well our models work"""
        print("\n EVALUATING MODEL PERFORMANCE")
        print("=" * 40)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n Testing {name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            self.predictions[name] = y_pred
            
            # Calculate performance metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape
            }
            
            print(f"  Mean Absolute Error: ${mae:.2f}")
            print(f"  Root Mean Square Error: ${rmse:.2f}")
            print(f"  R² Score: {r2:.4f}")
            print(f"  Mean Absolute Percentage Error: {mape:.2f}%")
            
            # Simple interpretation
            if r2 > 0.8:
                print(" Excellent performance!")
            elif r2 > 0.6:
                print(" Good performance!")
            elif r2 > 0.4:
                print(" Okay performance")
            else:
                print(" Needs improvement")
        
        return results
    
    def visualize_predictions(self, y_test, test_dates):
        """Create charts showing actual vs predicted prices"""
        print("\n CREATING PREDICTION VISUALIZATIONS")
        print("=" * 40)
        
        plt.figure(figsize=(14, 8))
        
        # Plot actual prices
        plt.plot(test_dates, y_test.values, label='Actual Prices', 
                linewidth=3, color='black', alpha=0.8)
        
        # Plot predictions for each model
        colors = ['red', 'blue', 'green']
        for i, (name, pred) in enumerate(self.predictions.items()):
            plt.plot(test_dates, pred, label=f'{name} Predictions', 
                    linewidth=2, alpha=0.7, color=colors[i % len(colors)])
        
        plt.title('Actual vs Predicted AAPL Stock Prices', fontsize=16, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("✅ Visualization created!")

# Test the model trainer
if __name__ == "__main__":
    print(" STEP 3: MODEL TRAINING AND EVALUATION")
    print("=" * 50)
    
    # Create predictor
    predictor = SimpleStockPredictor()
    
    # Prepare data
    data = predictor.prepare_data("AAPL", "5y")  
    
    # Create features and split data
    X_train, X_test, y_train, y_test = predictor.create_features_and_target()
    
    # Train models
    predictor.train_linear_model(X_train, y_train)
    predictor.train_random_forest(X_train, y_train)
    
    # Evaluate models
    results = predictor.evaluate_models(X_test, y_test)
    
    # Create visualizations
    test_start = len(data) - len(y_test)
    test_dates = data['Date'].iloc[test_start:].reset_index(drop=True)
    predictor.visualize_predictions(y_test, test_dates)
    
    # Summary
    print("\n MODEL TRAINING COMPLETED!")
    print("=" * 40)
    print(" Summary of Results:")
    for name, metrics in results.items():
        print(f"  {name}: R² = {metrics['R²']:.3f}, MAPE = {metrics['MAPE']:.1f}%")
    
    print("\n✅ Step 3 completed successfully!")