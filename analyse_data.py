# analyse_data.py
from data_loader import fetch_and_filter_markets

def run_dataframe_analysis():
    # Receive pre-filtered data directly
    df = fetch_and_filter_markets(query="AAPL", min_liquidity=1000.0)

    if df.empty:
        print("No market data returned.")
        return

    print("=" * 60)
    print("📈 MARKET SUMMARY METRICS")
    print("=" * 60)
    print(f"Total Active Markets:   {len(df)}")
    print(f"Total Market Liquidity: ${df['Pool Liquidity ($)'].sum():,.2f}")
    print(f"Total 24h Volume:       ${df['24h Volume ($)'].sum():,.2f}\n")

    print("=" * 60)
    print("TOP MARKETS BY LIQUIDITY")
    print("=" * 60)
    top_liquidity = df.sort_values(by="Pool Liquidity ($)", ascending=False).head(4)
    
    for idx, row in top_liquidity.iterrows():
        print(f"📌 Market: {row['Market Question']}")
        print(f"   • Liquidity:       ${row['Pool Liquidity ($)']:,.2f}")
        print(f"   • Implied Odds:    {row['Implied Odds (%)']:.1f}%")
        print(f"   • Liquidity Share: {row['Liquidity Share (%)']:.2f}%\n")

if __name__ == "__main__":
    run_dataframe_analysis()