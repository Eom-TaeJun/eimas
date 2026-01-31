
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def check_gold_data():
    ticker = 'GC=F'
    print(f"Checking data for {ticker}...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"Fetching from {start_date.date()} to {end_date.date()}")
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if df.empty:
            print("❌ No data found!")
            return
            
        print(f"✅ Data found: {len(df)} rows")
        print("\nLast 5 rows:")
        print(df.tail())
        
        last_date = df.index[-1]
        last_price = df['Close'].iloc[-1]
        
        print(f"\nLast data point date: {last_date}")
        print(f"Last price: {last_price}")
        
        days_diff = (end_date - last_date).days
        if days_diff > 3:
             print(f"⚠️ Warning: Data is {days_diff} days old!")
        else:
             print("✅ Data is recent.")
             
    except Exception as e:
        print(f"❌ Error fetching data: {e}")

if __name__ == "__main__":
    check_gold_data()
