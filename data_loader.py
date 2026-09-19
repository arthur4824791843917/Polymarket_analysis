# data_loader.py
import json
import urllib.parse
import urllib.request
import pandas as pd

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


def fetch_and_filter_markets(query="AAPL", min_liquidity=1000.0):
    """
    Handles API fetching AND filtering in one single place.
    Returns a clean Pandas DataFrame.
    """
    encoded_query = urllib.parse.quote(query)
    url = f"{GAMMA_API}/public-search?q={encoded_query}&events_status=active"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urllib.request.urlopen(req) as response:
            events = json.loads(response.read().decode()).get("events", [])
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

    market_records = []

    for event in events:
        event_title = event.get("title", "Untitled Event")
        event_desc = event.get("description", "")

        # Event-level filtering
        event_text = f"{event_title} {event_desc}".lower()
        if "market cap" in event_text or "valuation" in event_text:
            continue

        for market in event.get("markets", []):
            if market.get("closed") or not market.get("active", True):
                continue

            question = market.get("question") or market.get("groupItemTitle") or event_title
            if "market cap" in question.lower() or "valuation" in question.lower():
                continue

            pool_liq = float(market.get("liquidity", 0))
            if pool_liq < min_liquidity:
                continue

            raw_prices = market.get("outcomePrices", "[]")
            prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
            parsed_prices = [float(p) for p in prices if p is not None]

            # Exclude resolved/settled markets
            if any(p >= 0.999 or p <= 0.001 for p in parsed_prices):
                continue

            yes_price = parsed_prices[0] if len(parsed_prices) > 0 else 0.0
            no_price = parsed_prices[1] if len(parsed_prices) > 1 else 0.0
            vol_24h = float(market.get("volume24hr", 0))

            raw_tokens = market.get("clobTokenIds", "[]")
            tokens = json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens

            # Extract end date from the API market object
            end_date_str = market.get("endDate") or market.get("endDateIso", "N/A")

            market_records.append({
                "Event": event_title,
                "Market Question": question,
                "Closing Date": end_date_str,  # <--- FIXED: Added column to dictionary
                "Pool Liquidity ($)": pool_liq,
                "24h Volume ($)": vol_24h,
                "Yes Price ($)": yes_price,
                "No Price ($)": no_price,
                "Implied Odds (%)": round(yes_price * 100, 2),
                "Tokens": tokens
            })

    df = pd.DataFrame(market_records)
    
    # Calculate relative liquidity share once for all downstream scripts
    if not df.empty:
        total_liq = df["Pool Liquidity ($)"].sum()
        df["Liquidity Share (%)"] = (df["Pool Liquidity ($)"] / total_liq) * 100 if total_liq > 0 else 0.0
        
        # FIXED: Kept inside 'if not df.empty:' block to prevent errors on empty queries
        df["Closing Date"] = pd.to_datetime(df["Closing Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    return df


def fetch_orderbook_liquidity(token_id):
    """Utility to query orderbook depth if needed downstream."""
    url = f"{CLOB_API}/book?token_id={token_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as response:
            book = json.loads(response.read().decode())
            bids = sum(float(b.get("size", 0)) * float(b.get("price", 0)) for b in book.get("bids", []))
            asks = sum(float(a.get("size", 0)) * float(a.get("price", 0)) for a in book.get("asks", []))
            return bids + asks
    except Exception:
        return 0.0