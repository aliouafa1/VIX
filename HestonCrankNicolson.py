import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xbbg import blp
import datetime

def get_bloomberg_data(tickers, start_date, end_date):
    """
    Fetches Bloomberg intraday price data for the given tickers.
    Returns a dataframe with adjusted timestamps.
    """
    df = blp.bdh(
        tickers=tickers, 
        flds=["PX_LAST"], 
        start_date=start_date, 
        end_date=end_date,
        fill="prev"
    )
    
    df = df.stack().reset_index()
    df.columns = ["Date", "Ticker", "Close"]
    
    return df.pivot(index="Date", columns="Ticker", values="Close")

def compute_realized_volatility(returns, period):
    """
    Computes rolling realized volatility for a given period.
    """
    return returns.rolling(period).std() * np.sqrt(252)

def get_market_holidays():
    """
    Returns a set of market holidays for both US and Hong Kong.
    Bloomberg provides holiday calendars via 'BDAYS' but here, we assume manual control.
    """
    us_holidays = set(pd.to_datetime([
        "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27", 
        "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25"
    ]))  # Add actual US market holidays

    hk_holidays = set(pd.to_datetime([
        "2024-01-01", "2024-02-10", "2024-02-12", "2024-04-04", "2024-05-01",
        "2024-06-10", "2024-07-01", "2024-10-01", "2024-12-25"
    ]))  # Add actual HK market holidays

    return us_holidays, hk_holidays

def adjust_for_holidays(data):
    """
    Removes weekends and ensures holidays leave blank rows.
    """
    us_holidays, hk_holidays = get_market_holidays()
    
    full_date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='B')
    
    # Remove holidays
    full_date_range = [date for date in full_date_range if date not in us_holidays and date not in hk_holidays]
    
    data = data.reindex(full_date_range)
    
    return data

def main():
    # User input parameters
    period = int(input("Enter the RV computation period (e.g., 21 for 1-month rolling RV): "))

    # Define tickers
    tickers = ["FXI US Equity", "HC1 Index"]
    
    # Define date range
    start_date = "2023-01-01"
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    
    # Fetch data
    price_data = get_bloomberg_data(tickers, start_date, end_date)
    
    # Compute log returns
    returns = np.log(price_data / price_data.shift(1))
    
    # Compute realized volatility
    rv = compute_realized_volatility(returns, period)
    
    # Compute RV spread
    rv["RV_Spread"] = rv["FXI US Equity"] - rv["HC1 Index"]
    
    # Adjust for holidays and weekends
    rv = adjust_for_holidays(rv)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(rv.index, rv["FXI US Equity"], label="FXI RV", linestyle="--")
    plt.plot(rv.index, rv["HC1 Index"], label="HC1 RV", linestyle="--")
    plt.plot(rv.index, rv["RV_Spread"], label="RV Spread", linewidth=2)
    
    plt.legend()
    plt.title("Realized Volatility Spread: FXI US vs HC1 Index")
    plt.xlabel("Date")
    plt.ylabel("Realized Volatility")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
