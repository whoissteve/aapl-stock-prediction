# Simple test to make sure everything works
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

print("🚀 Starting basic stock data test...")

# Download just 1 month of data to test
print(" Downloading Apple stock data...")
stock = yf.Ticker("AAPL")
data = stock.history(period="1mo")

print(f"✅ Downloaded {len(data)} days of data")
print(f" Date range: {data.index[0].date()} to {data.index[-1].date()}")
print(f" Latest closing price: ${data['Close'][-1]:.2f}")

# Show first few rows
print("\n First 5 rows of data:")
print(data.head())

print("\n Basic test completed successfully!")