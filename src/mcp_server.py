from mcp.server.mcpserver import MCPServer
import xgboost as xgb
from data_loader import load_etf_data
from news_rag import query_news_db

mcp = MCPServer("ETF Intelligence System")

@mcp.tool()
def predict_etf_movement(ticker: str = "SPY") -> str:
    """
    Predict the next day's price movement for a given ETF ticker.
    Returns 'Up' if the price is predicted to go up, 'Down' otherwise.
    """
    # Load ETF data for the specified ticker
    X_train, X_test, y_train, y_test = load_etf_data(ticker)

    # Train the XGBoost model
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # Make prediction for the last available data point
    latest_features = X_test.iloc[[-1]]
    prediction = model.predict(latest_features)[0]

    direction = "Up" if prediction == 1 else "Down"
    return f"Quantitative Model Prediction for {ticker}: {direction}"

@mcp.tool()
def get_market_news(query: str, ticker: str = "SPY", top_k: int = 3) -> str:
    """
    Retrieve the latest market news related to a given ETF ticker.
    Returns the top_k news articles as a string.
    """
    results = query_news_db(
        query=query,
        ticker_symbol=ticker,
        top_k=top_k
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not documents:
        return f"No recent news found for query '{query}' (ticker {ticker})."

    formatted_outputs = (
        f"Top {top_k} news articles for "
        f"query '{query}' (ticker {ticker}):\n\n"
    )

    for i, (doc, meta) in enumerate(
        zip(documents, metadatas),
        start=1
    ):
        formatted_outputs += f"{i}. **{meta.get('title')}**\n"
        formatted_outputs += (
            f"   Publisher: {meta.get('publisher')} | "
            f"Published: {meta.get('published_time')}\n"
        )
        formatted_outputs += f"   Summary/Context: {doc}\n\n"

    return formatted_outputs

if __name__ == "__main__":
    mcp.run()