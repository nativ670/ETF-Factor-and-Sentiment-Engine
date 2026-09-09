def engineer_features(data):
    """
    Engineer features for the given ETF data.
    """
    data['Daily_Return'] = data['Price'].pct_change()  # Daily return
    data['Rolling_Vol_14'] = data['Daily_Return'].rolling(window=14).std()  # 14-day rolling volatility
    data['SMA_5_Ratio'] = data['Price'] / data['Price'].rolling(window=5).mean()  # 5-day SMA ratio
    data['SMA_10_Ratio'] = data['Price'] / data['Price'].rolling(window=10).mean()  # 10-day SMA ratio
    data['SMA_50_Ratio'] = data['Price'] / data['Price'].rolling(window=50).mean()  # 50-day SMA ratio
    data['Rolling_Std_5'] = data['Daily_Return'].rolling(window=5).std()  # 5-day rolling std deviation
    data['Rolling_Std_10'] = data['Daily_Return'].rolling(window=10).std()  # 10-day rolling std deviation
    data.dropna(inplace=True)  # Drop rows with missing values after feature engineering
    return data