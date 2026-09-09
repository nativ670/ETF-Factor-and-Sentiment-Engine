import yfinance

def load_etf_data(symbol, start="2020-01-01", end="2026-01-01", period="1d"):
    """
    Load historical data for a given ETF symbol using yfinance.
    """
    etf = yfinance.Ticker(symbol)
    data = etf.history(start=start, end=end)
    target_column = "Close"  # You can change this to "Open", "High", "Low", etc. if needed
    data = data[[target_column]].rename(columns={target_column: "Price"})
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].dt.date  # Convert to date only
    data["Next_Day_Price"] = data["Price"].shift(-1)  # Create a new column for the next day's price
    data.dropna(inplace=True)  # Drop rows with missing values
    data['Target'] = (data["Next_Day_Price"] > data["Price"]).astype(int)  # Create a binary target variable
    train_size = int(len(data) * 0.8)
    test_size = len(data) - train_size
    X = data[["Price"]]
    y = data["Target"]
    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:train_size + test_size]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:train_size + test_size]
    return X_train, X_test, y_train, y_test