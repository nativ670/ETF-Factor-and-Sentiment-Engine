import chromadb
import yfinance


DB_PATH = "./chroma_db"
COLLECTION_NAME = "financial_news"


def get_collection():
    """Get or create the persistent ChromaDB news collection."""
    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def update_news_db(ticker_symbol: str = "SPY"):
    """
    Fetch news for a ticker and idempotently upsert it into ChromaDB.
    """

    collection = get_collection()

    ticker = yfinance.Ticker(ticker_symbol)
    news_items = ticker.news

    if not news_items:
        print(f"No news found for ticker: {ticker_symbol}")
        return

    documents = []
    metadatas = []
    ids = []

    for news_item in news_items:

        article_id = news_item.get("id")

        if not article_id:
            continue

        content = news_item.get("content", {})

        if not isinstance(content, dict):
            continue

        title = content.get("title", "")

        provider = content.get("provider", {})

        if isinstance(provider, dict):
            publisher = provider.get("displayName", "Unknown")
        else:
            publisher = str(provider) if provider else "Unknown"

        click_url = content.get("clickThroughUrl") or {}

        if isinstance(click_url, dict):
            link = click_url.get("url", "")
        else:
            link = ""

        published_time = content.get("pubDate", "")
        summary = content.get("summary", "")

        combined_document = (
            f"Title: {title} | "
            f"Publisher: {publisher} | "
            f"Published Time: {published_time} | "
            f"Summary: {summary}"
        )

        documents.append(combined_document)

        metadatas.append({
            "title": title,
            "publisher": str(publisher),
            "link": link,
            "published_time": str(published_time),
            "ticker": ticker_symbol
        })

        ids.append(str(article_id))

    if not ids:
        print("No valid news articles found.")
        return

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(
        f"Successfully upserted "
        f"{len(ids)} news articles for {ticker_symbol}."
    )


def query_news_db(
    query: str,
    top_k: int = 5,
    ticker_symbol: str | None = None
):
    """
    Search the news database semantically.

    Optionally restrict results to a specific ticker.
    """

    collection = get_collection()

    where_filter = None

    if ticker_symbol:
        where_filter = {
            "ticker": ticker_symbol
        }

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_filter
    )

    return results


if __name__ == "__main__":

    print("Updating news database...")
    update_news_db("SPY")

    print("\nQuerying database...")

    results = query_news_db(
        "market trends",
        top_k=5,
        ticker_symbol="SPY"
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        print("No relevant articles found.")

    else:
        print(f"Found {len(documents)} relevant articles:\n")

        for i, (document, metadata) in enumerate(
            zip(documents, metadatas), 1
        ):
            print(f"{i}. {metadata['title']}")
            print(f"   Publisher: {metadata['publisher']}")
            print(f"   Published: {metadata['published_time']}")
            print(f"   Link: {metadata['link']}")
            print()