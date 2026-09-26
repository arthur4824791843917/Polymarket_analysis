import json
import re
import urllib.parse
import urllib.request
import pandas as pd
from datetime import datetime, timedelta

GAMMA_API = "https://gamma-api.polymarket.com"


def extract_price_condition(question):
    q = question.strip()

    # 1. Price ranges: e.g. "$315-$320", "$315 – $320", "$315—$320"
    m_range = re.search(r'\$(\d+(?:\.\d+)?)\s*[-–—]\s*\$?(\d+(?:\.\d+)?)', q)
    if m_range:
        low, high = float(m_range.group(1)), float(m_range.group(2))
        return f"${low:.0f}-${high:.0f}", low, high, "range"

    # 2. Operators: e.g. ">$355", "<$310", "≥$300"
    m_op = re.search(r'([><≥≤]=?)\s*\$?(\d+(?:\.\d+)?)', q)
    if m_op:
        op, val = m_op.group(1), float(m_op.group(2))
        if '>' in op or '≥' in op:
            return f">${val:.0f}+", val, None, "above"
        elif '<' in op or '≤' in op:
            return f"<{val:.0f}", None, val, "below"

    # 3. Text direction: e.g. "close above $350", "hit (HIGH) $352"
    m_desc = re.search(r'\b(above|below|over|under|hit \(high\)|hit \(low\))\s+\$?(\d+(?:\.\d+)?)', q, re.IGNORECASE)
    if m_desc:
        direction, val = m_desc.group(1).lower(), float(m_desc.group(2))
        if any(kw in direction for kw in ['above', 'over', 'high']):
            return f">${val:.0f}+", val, None, "above"
        elif any(kw in direction for kw in ['below', 'under', 'low']):
            return f"<{val:.0f}", None, val, "below"

    # 4. Fallback single price match: e.g. "$350"
    m_single = re.search(r'\$(\d+(?:\.\d+)?)', q)
    if m_single:
        val = float(m_single.group(1))
        return f"${val:.0f}", val, val, "exact"

    return "N/A", None, None, "unknown"


def extract_event_window(question, closing_date_str=None):
    def fmt(dt):
        return dt.strftime('%d-%m-%Y') if pd.notna(dt) else None

    q = question.strip()
    year = datetime.now().year
    if closing_date_str and str(closing_date_str) != 'nan':
        try:
            year = pd.to_datetime(closing_date_str).year
        except Exception:
            pass

    # Match week ranges
    m_range = re.search(r'(?:week of\s+)?([A-Za-z]+)\s+(\d{1,2})\s*[-–—]\s*(?:([A-Za-z]+)\s+)?(\d{1,2})(?:\s+(\d{4}))?', q, re.IGNORECASE)
    if m_range:
        m1, d1, m2, d2, yr = m_range.groups()
        yr = int(yr) if yr else year
        m2 = m2 if m2 else m1
        try:
            start_dt = pd.to_datetime(f"{m1} {d1} {yr}")
            end_dt = pd.to_datetime(f"{m2} {d2} {yr}")
            return fmt(start_dt), fmt(end_dt)
        except Exception:
            pass

    # Match single dates
    m_day = re.search(r'\bon\s+([A-Za-z]+)\s+(\d{1,2})(?:\s+(\d{4}))?', q, re.IGNORECASE)
    if m_day:
        month, day, yr = m_day.groups()
        yr = int(yr) if yr else year
        try:
            dt = pd.to_datetime(f"{month} {day} {yr}")
            return fmt(dt), fmt(dt)
        except Exception:
            pass

    if closing_date_str and str(closing_date_str) != 'nan':
        try:
            c_dt = pd.to_datetime(closing_date_str)
            return fmt(c_dt), fmt(c_dt)
        except Exception:
            pass

    return None, None


def fetch_and_filter_markets(query="AAPL", min_liquidity=0.0):
    """
    Directly queries Gamma public search endpoint for active events matching the term.
    """
    encoded_query = urllib.parse.quote(query)
    url = f"{GAMMA_API}/public-search?q={encoded_query}&events_status=active"
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            events = data.get("events", []) if isinstance(data, dict) else []
    except Exception as e:
        print(f"API Request Error: {e}")
        return pd.DataFrame()

    market_records = []

    for event in events:
        event_title = event.get("title", "Untitled Event")
        
        for market in event.get("markets", []):
            if market.get("closed") or not market.get("active", True):
                continue

            question = market.get("question") or market.get("groupItemTitle") or event_title
            pool_liq = float(market.get("liquidity", 0))

            if pool_liq < min_liquidity:
                continue

            raw_prices = market.get("outcomePrices", "[]")
            prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
            parsed_prices = [float(p) for p in prices if p is not None]

            # Drop resolved/settled outcomes
            if any(p >= 0.999 or p <= 0.001 for p in parsed_prices):
                continue

            yes_price = parsed_prices[0] if len(parsed_prices) > 0 else 0.0
            no_price = parsed_prices[1] if len(parsed_prices) > 1 else 0.0
            vol_24h = float(market.get("volume24hr", 0))

            raw_tokens = market.get("clobTokenIds", "[]")
            tokens = json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens

            end_date_str = (
                market.get("endDate")
                or market.get("endDateIso")
                or market.get("resolutionTime")
                or event.get("endDate")
            )
            closing_date = pd.to_datetime(end_date_str, errors="coerce").strftime("%d-%m-%Y") if end_date_str else None

            start_date, end_date = extract_event_window(question, closing_date)
            price_target, min_price, max_price, cond_type = extract_price_condition(question)

            market_records.append({
                "Event": event_title,
                "Market Question": question,
                "Price Target": price_target,
                "Min Price Target": min_price,
                "Max Price Target": max_price,
                "Condition Type": cond_type,
                "Start Date": start_date,
                "End Date": end_date,
                "Closing Date": closing_date,
                "Pool Liquidity ($)": pool_liq,
                "24h Volume ($)": vol_24h,
                "Yes Price ($)": yes_price,
                "No Price ($)": no_price,
                "Implied Odds (%)": round(yes_price * 100, 2),
                "Tokens": tokens
            })

    df = pd.DataFrame(market_records)

    if not df.empty:
        total_liq = df["Pool Liquidity ($)"].sum()
        df["Liquidity Share (%)"] = (df["Pool Liquidity ($)"].div(total_liq) * 100) if total_liq > 0 else 0.0

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(f"Retrieved {len(df)} markets for query '{query}':")
    print(df)
    return df


if __name__ == "__main__":
    fetch_and_filter_markets(query="AAPL", min_liquidity=0.0)