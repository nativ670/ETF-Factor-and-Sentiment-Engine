import hashlib
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import chromadb
import yfinance

# -------------------------------------------------------------------
# Configuration & Absolute Path Setup
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "financial_news"

NEWS_SOURCE_TICKERS = [
    # Broad market
    "SPY", "QQQ", "DIA", "IWM",
    # Sectors
    "XLE", "XLK", "XLF", "XLV",
    "XLI", "XLP", "XLY",
    # Commodities
    "GLD", "SLV", "USO",
    # Bonds / international
    "TLT", "EEM",
    # Major technology
    "NVDA", "AAPL", "MSFT", "GOOGL",
    "AMZN", "META", "AVGO", "AMD",
    "TSM", "INTC",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    "OXY", "EOG",
    # Financials
    "JPM", "BAC", "GS", "MS",
    # Healthcare
    "LLY", "JNJ", "PFE", "UNH",
    # Industrials
    "CAT", "GE", "HON", "BA",
    # Consumer
    "WMT", "COST", "HD", "MCD"
]

# -------------------------------------------------------------------
# Persistent Client Singleton
# -------------------------------------------------------------------
_CLIENT = None
_COLLECTION = None

def get_collection():
    """Returns a singleton ChromaDB client collection to prevent lock contention."""
    global _CLIENT, _COLLECTION
    if _COLLECTION is None:
        _CLIENT = chromadb.PersistentClient(path=DB_PATH)
        _COLLECTION = _CLIENT.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _COLLECTION


# -------------------------------------------------------------------
# Ticker Metadata Profiling (Cached)
# -------------------------------------------------------------------
@lru_cache(maxsize=128)
def get_ticker_context(ticker_symbol: str) -> str:
    """
    Cached company summary lookups to eliminate network latency
    during semantic queries.
    """
    ticker = yfinance.Ticker(ticker_symbol)
    try:
        info = ticker.info
        name = info.get("longName", "")
        sector = info.get("sector", "")
        industry = info.get("industry", "")
        description = info.get("longBusinessSummary", "")
        return (
            f"{ticker_symbol}. Name: {name}. Sector: {sector}. "
            f"Industry: {industry}. Description: {description}"
        )
    except Exception as e:
        print(f"Could not retrieve context for {ticker_symbol}: {e}")
        return ticker_symbol


# -------------------------------------------------------------------
# Ingestion & Normalization
# -------------------------------------------------------------------
def update_news_db(ticker_symbol: str):
    """
    Fetches news for one source ticker and upserts it into ChromaDB.
    """
    collection = get_collection()
    print(f"Fetching news for {ticker_symbol}...")

    ticker = yfinance.Ticker(ticker_symbol)
    try:
        news_items = ticker.news
    except Exception as e:
        print(f"Could not retrieve news for {ticker_symbol}: {e}")
        return

    if not news_items:
        print(f"No news found for {ticker_symbol}")
        return

    documents = []
    metadatas = []
    ids = []

    for news_item in news_items:
        article_id = news_item.get("id")
        content = news_item.get("content", {})
        if not isinstance(content, dict):
            continue

        title = content.get("title", "").strip()
        provider = content.get("provider", {})
        if isinstance(provider, dict):
            publisher = provider.get("displayName", "Unknown")
        else:
            publisher = str(provider) if provider else "Unknown"

        click_url = content.get("clickThroughUrl") or {}
        link = click_url.get("url", "") if isinstance(click_url, dict) else ""
        published_time = content.get("pubDate", "")
        summary = content.get("summary", "").strip()

        # Fallback ID generation to avoid empty string collisions
        if not article_id:
            raw_key = link or f"{title}_{published_time}"
            if not raw_key:
                continue
            article_id = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

        combined_document = f"{title}. {summary}"

        documents.append(combined_document)
        metadatas.append({
            "title": title,
            "publisher": str(publisher),
            "link": link,
            "published_time": str(published_time),
            "source_ticker": ticker_symbol,
        })
        ids.append(str(article_id))

    if ids:
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Upserted {len(ids)} articles from {ticker_symbol}.")


def update_global_news_db(delay_seconds: float = 0.2):
    """
    Iterates through market sources with slight throttling to prevent 429 errors.
    """
    print("\nUpdating global news database...\n")
    total = 0

    for ticker in NEWS_SOURCE_TICKERS:
        before = get_collection().count()
        update_news_db(ticker)
        after = get_collection().count()
        added = after - before
        total += max(added, 0)
        time.sleep(delay_seconds)

    final_count = get_collection().count()
    print(
        f"\nGlobal news database update complete.\n"
        f"Articles added during this update: {total}\n"
        f"Total articles in database: {final_count}\n"
    )


# -------------------------------------------------------------------
# Scoring & Ranking Pipeline
# -------------------------------------------------------------------
def parse_published_time(published_time: str):
    if not published_time:
        return None
    try:
        return datetime.fromisoformat(published_time.replace("Z", "+00:00"))
    except Exception:
        return None


def calculate_recency_score(published_time: str) -> float:
    published_dt = parse_published_time(published_time)
    if published_dt is None:
        return 0.0
    now = datetime.now(timezone.utc)
    age_hours = max((now - published_dt).total_seconds() / 3600, 0)
    return 1 / (1 + age_hours / 24)


def query_news_db(query: str, top_k: int = 5, ticker_symbol: str | None = None):
    """
    Symmetrical dual-query search combining topic relevance, ticker relevance,
    and recency decay.
    """
    collection = get_collection()
    collection_count = collection.count()

    if collection_count == 0:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "scores": [[]]
        }

    candidate_k = min(max(top_k * 10, 30), collection_count)

    # 1. Query by topic
    topic_results = collection.query(
        query_texts=[query],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"]
    )

    # 2. Query by ticker (independent candidate generator)
    ticker_results = None
    if ticker_symbol:
        ticker_context = get_ticker_context(ticker_symbol)
        ticker_query = f"Security: {ticker_symbol}. {ticker_context}"
        ticker_results = collection.query(
            query_texts=[ticker_query],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"]
        )

    candidates = {}

    def extract_article_key(metadata, document):
        return metadata.get("link") or metadata.get("title") or hashlib.md5(document.encode()).hexdigest()

    # Process topic matches
    for doc, meta, dist in zip(
        topic_results["documents"][0],
        topic_results["metadatas"][0],
        topic_results["distances"][0]
    ):
        key = extract_article_key(meta, doc)
        candidates[key] = {
            "document": doc,
            "metadata": meta,
            "topic_distance": dist,
            "ticker_distance": 1.0  # default baseline
        }

    # Merge ticker matches symmetrically
    if ticker_results:
        for doc, meta, dist in zip(
            ticker_results["documents"][0],
            ticker_results["metadatas"][0],
            ticker_results["distances"][0]
        ):
            key = extract_article_key(meta, doc)
            if key in candidates:
                candidates[key]["ticker_distance"] = dist
            else:
                candidates[key] = {
                    "document": doc,
                    "metadata": meta,
                    "topic_distance": 1.0,
                    "ticker_distance": dist
                }

    # 3. Score candidates
    scored_articles = []
    for article in candidates.values():
        # Cosine distance to similarity: clamp between 0.0 and 1.0
        topic_sim = max(0.0, min(1.0, 1.0 - article["topic_distance"]))
        ticker_sim = max(0.0, min(1.0, 1.0 - article["ticker_distance"]))
        recency = calculate_recency_score(article["metadata"].get("published_time", ""))

        if ticker_symbol:
            final_score = (0.50 * topic_sim) + (0.35 * ticker_sim) + (0.15 * recency)
        else:
            final_score = (0.85 * topic_sim) + (0.15 * recency)

        article.update({
            "topic_similarity": topic_sim,
            "ticker_similarity": ticker_sim,
            "recency_score": recency,
            "final_score": final_score
        })
        scored_articles.append(article)

    # Sort descending by composite score
    scored_articles.sort(key=lambda x: x["final_score"], reverse=True)
    scored_articles = scored_articles[:top_k]

    return {
        "documents": [[a["document"] for a in scored_articles]],
        "metadatas": [[a["metadata"] for a in scored_articles]],
        "distances": [[a["topic_distance"] for a in scored_articles]],
        "scores": [[
            {
                "topic_similarity": a["topic_similarity"],
                "ticker_similarity": a["ticker_similarity"],
                "recency_score": a["recency_score"],
                "final_score": a["final_score"]
            }
            for a in scored_articles
        ]]
    }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING RETRIEVAL PIPELINE")
    print("=" * 70)

    results = query_news_db(
        query="AI semiconductor chips demand",
        top_k=3,
        ticker_symbol="NVDA"
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    scores = results.get("scores", [[]])[0]

    if not docs:
        print("No articles returned. Run update_news_db('NVDA') or update_global_news_db() to populate.")
    else:
        for i, (meta, score) in enumerate(zip(metas, scores), 1):
            print(f"{i}. {meta.get('title')}")
            print(f"   Publisher: {meta.get('publisher')}")
            print(f"   Topic Sim: {score['topic_similarity']:.3f} | Ticker Sim: {score['ticker_similarity']:.3f} | Recency: {score['recency_score']:.3f}")
            print(f"   Composite: {score['final_score']:.3f}\n")