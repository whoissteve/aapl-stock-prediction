# Step 2: Data Cleaning and Feature Engineering - FIXED VERSION
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

class StockDataCleaner:
    def __init__(self):
        self.raw_data = None
        self.processed_data = None  # Changed from clean_data to processed_data
        
    def load_data(self, symbol="AAPL", period="5y"):
        """Load fresh data"""
        print(f" Loading {symbol} data...")
        stock = yf.Ticker(symbol)
        self.raw_data = stock.history(period=period)
        print(f"✅ Loaded {len(self.raw_data)} records")
        
    def clean_data(self):
        """Clean the data - handle missing values and outliers"""
        print("\n🧹 CLEANING DATA")
        print("=" * 40)
        
        # Start with a copy to make a date column
        df = self.raw_data.copy()
        df.reset_index(inplace=True)  
        
        print(f" Original data shape: {df.shape}")
        
        # Check for missing values
        missing_before = df.isnull().sum().sum()
        print(f"🔍 Missing values before cleaning: {missing_before}")
        
        # Fill missing values
        df.fillna(method='ffill', inplace=True) 
        df.dropna(inplace=True) 
        # Remove duplicates
        duplicates = df.duplicated().sum()
        df.drop_duplicates(inplace=True)
        print(f"  Removed {duplicates} duplicate records")
        
        # Handle outliers in Close price
        Q1 = df['Close'].quantile(0.25)
        Q3 = df['Close'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df['Close'] < lower_bound) | (df['Close'] > upper_bound)]
        print(f"  Found {len(outliers)} outliers in Close price")
        
        # Cap outliers instead of removing them
        df['Close'] = df['Close'].clip(lower=lower_bound, upper=upper_bound)
        
        missing_after = df.isnull().sum().sum()
        print(f"✅ Missing values after cleaning: {missing_after}")
        print(f" Final data shape: {df.shape}")
        
        self.processed_data = df  # Changed from self.clean_data
        return df
    
    def add_basic_features(self):
        """Add simple technical indicators"""
        print("\n🔧 ADDING TECHNICAL INDICATORS")
        print("=" * 40)
        
        df = self.processed_data.copy()  # Changed from self.clean_data
        
        # Moving averages
        print(" Adding moving averages...")
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # Price changes
        print(" Adding price change indicators...")
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_Abs'] = df['Price_Change'].abs()
        
        # Volume indicators
        print(" Adding volume indicators...")
        df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_10']
        
        # Remove rows with NaN (created by rolling calculations)
        df.dropna(inplace=True)
        
        print(f"✅ Added technical indicators. New shape: {df.shape}")
        print(f" New columns: {[col for col in df.columns if col not in self.processed_data.columns]}")
        
        self.processed_data = df  # Changed from self.clean_data
        return df
    
    def show_summary(self):
        """Show summary of cleaned data"""
        if self.processed_data is None:  # Changed from self.clean_data
            print("❌ No processed data available")
            return
            
        print("\n PROCESSED DATA SUMMARY")
        print("=" * 40)
        print(f"Shape: {self.processed_data.shape}")
        print(f"Date range: {self.processed_data['Date'].min().date()} to {self.processed_data['Date'].max().date()}")
        
        print("\n PRICE STATISTICS")
        print("=" * 40)
        print(self.processed_data[['Close', 'MA_5', 'MA_10', 'MA_20']].describe())
        
        # Create visualization
        print("\n Creating visualization...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Price chart
        ax1.plot(self.processed_data['Date'], self.processed_data['Close'], label='Close Price', linewidth=2)
        ax1.plot(self.processed_data['Date'], self.processed_data['MA_20'], label='20-day MA', alpha=0.7)
        ax1.set_title('AAPL Price with Moving Average')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True)
        
        # Volume chart
        ax2.plot(self.processed_data['Date'], self.processed_data['Volume'], color='orange', alpha=0.7)
        ax2.set_title('Trading Volume')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# Test the cleaner
if __name__ == "__main__":
    print(" STEP 2: DATA CLEANING & FEATURE ENGINEERING")
    print("=" * 60)
    
    # Create cleaner and process data
    cleaner = StockDataCleaner()
    cleaner.load_data("AAPL", "5y")
    clean_data = cleaner.clean_data()
    featured_data = cleaner.add_basic_features()
    cleaner.show_summary()
    
    print("\n Step 2 completed successfully!")