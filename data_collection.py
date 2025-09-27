# Step 1: Data Collection and Basic Cleaning
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

class StockDataFetcher:
    def __init__(self, symbol="AAPL", period="5y"):
        self.symbol = symbol
        self.period = period
        self.raw_data = None
        
    def fetch_data(self):
        """Download historical stock data"""
        print(f"🔄 Fetching {self.period} of {self.symbol} data...")
        
        try:
            stock = yf.Ticker(self.symbol)
            self.raw_data = stock.history(period=self.period)
            
            if self.raw_data.empty:
                raise ValueError(f"No data found for {self.symbol}")
                
            print(f"✅ Success! Downloaded {len(self.raw_data)} records")
            print(f"📅 From {self.raw_data.index[0].date()} to {self.raw_data.index[-1].date()}")
            
            return self.raw_data
            
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    def basic_info(self):
        """Show basic information about the data"""
        if self.raw_data is None:
            print("❌ No data available. Run fetch_data() first.")
            return
            
        print("\n DATA OVERVIEW")
        print("=" * 40)
        print(f"Shape: {self.raw_data.shape}")
        print(f"Columns: {list(self.raw_data.columns)}")
        print(f"Date range: {self.raw_data.index[0].date()} to {self.raw_data.index[-1].date()}")
        
        print("\n PRICE SUMMARY")
        print("=" * 40)
        print(self.raw_data['Close'].describe())
        
        print("\n MISSING VALUES")
        print("=" * 40)
        missing = self.raw_data.isnull().sum()
        print(missing)
        
        print("\n FIRST 5 ROWS")
        print("=" * 40)
        print(self.raw_data.head())
        
        print("\n LAST 5 ROWS") 
        print("=" * 40)
        print(self.raw_data.tail())

# Test the data fetcher
if __name__ == "__main__":
    print(" STEP 1: DATA COLLECTION")
    print("=" * 50)
    
    # Create fetcher and download data
    fetcher = StockDataFetcher("AAPL", "5y")
    data = fetcher.fetch_data()
    fetcher.basic_info()
    
    # Simple visualization
    print("\n Creating basic price chart...")
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'])
    plt.title('AAPL Stock Price - Last 5 Years')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True)
    plt.show()
    
    print("\n Step 1 completed successfully!")