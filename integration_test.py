# Integration Test - Make sure everything works together
import os
import sys

print("INTEGRATION TEST - CHECKING ALL COMPONENTS")
print("=" * 60)

# Check all imports
print("\n 1 TESTING IMPORTS...")
try:
    import pandas as pd
    import yfinance as yf
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Check data download
print("\n 2 TESTING DATA DOWNLOAD...")
try:
    stock = yf.Ticker("AAPL")
    test_data = stock.history(period="5d")  # Just 5 days for quick test
    print(f"✅ Downloaded {len(test_data)} days of data")
except Exception as e:
    print(f"❌ Data download failed: {e}")

# Check basic data processing
print("\n 3 TESTING DATA PROCESSING...")
try:
    test_data['MA_5'] = test_data['Close'].rolling(5).mean()
    test_data.dropna(inplace=True)
    print(f"✅ Data processing works - {len(test_data)} rows after adding indicators")
except Exception as e:
    print(f"❌ Data processing failed: {e}")

# Check model training
print("\n 4 TESTING MODEL TRAINING...")
try:
    # Simple dummy data for model test
    X_dummy = np.random.random((100, 5))
    y_dummy = np.random.random(100)
    
    model = LinearRegression()
    model.fit(X_dummy, y_dummy)
    predictions = model.predict(X_dummy[:10])
    print(f"✅ Model training works - made {len(predictions)} predictions")
except Exception as e:
    print(f"❌ Model training failed: {e}")

# Check file structure
print("\n5️ CHECKING PROJECT STRUCTURE...")
expected_files = ['test_basic.py', 'data_collection.py', 
                 'data_cleaning.py', 'prediction_model.py']

for file in expected_files:
    if os.path.exists(file):
        print(f"✅ {file} exists")
    else:
        print(f"  {file} missing")

print("\n INTEGRATION TEST COMPLETED!")
print(" Ready to run the full pipeline!")