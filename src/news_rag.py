import chromadb
import yfinance


def update_news_db(ticker_symbol: str = "SPY"):
    """
    Fetches news for a given ETF ticker and performs an idempotent 
    batch upsert into the local ChromaDB vector database.
    """
    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_or_create_collection(name="financial_news", metadata={"hnsw:space": "cosine"}
    )

    etf = yfinance.Ticker(ticker_symbol)
    etf_news = etf.news

    if not etf_news:
        print(f"No news found for ticker: {ticker_symbol}")
        return

    documents = []
    metadatas = []
    ids = []

    for news_item in etf_news:
        article_id = news_item.get("uuid", "")
        if not article_id:
            continue  # Skip if there's no unique identifier for the news item

        ticker = news_item.get("ticker", "")
        title = news_item.get("title", "")
        publisher = news_item.get("publisher", "")
        link = news_item.get("link", "")
        published_time = news_item.get("providerPublishTime", 0)
        summary = news_item.get("summary", "")

        # Combine title and summary into a single text document for embedding
        combined_document = f"Title: {title} | Publisher: {publisher} | Published Time: {published_time} | Summary: {summary}"
        documents.append(combined_document)
        metadatas.append({"title": title, "publisher": publisher, "link": link, "published_time": published_time, "ticker": ticker})
        ids.append(article_id)

    # Insert the news items into the ChromaDB collection
    if ids:
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully upserted {len(ids)} news articles for {ticker_symbol} into ChromaDB.")

if __name__ == "__main__":
    update_news_db("SPY")  # Update news for SPY ETF
