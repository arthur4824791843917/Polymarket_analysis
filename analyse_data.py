# analyse_data.py
from data_loader import fetch_and_filter_markets

def run_dataframe_analysis():
    # Receive pre-filtered data directly
    df = fetch_and_filter_markets(query="AAPL", min_liquidity=1000.0)

    if df.empty:
        print("No market data returned.")
        return

    print("MARKET SUMMARY METRICS \n")
    print(f"Total Active Markets:   {len(df)}")
    print(f"Total Market Liquidity: ${df['Pool Liquidity ($)'].sum():,.2f}")
    print(f"Total 24h Volume:       ${df['24h Volume ($)'].sum():,.2f}\n \n \n \n")

    print("TOP MARKETS BY LIQUIDITY \n")
    top_liquidity = df.sort_values(by="Pool Liquidity ($)", ascending=False).head(16)
    
    for idx, row in top_liquidity.iterrows():
        print(f"📌 Market: {row['Market Question']}")
        print(f"    Liquidity:       ${row['Pool Liquidity ($)']:,.2f}")
        print(f"    Implied Odds:    {row['Implied Odds (%)']:.1f}%")
        print(f"    Liquidity Share: {row['Liquidity Share (%)']:.2f}%")
        print(f"    Closing date:    {row['Closing Date']}\n")

if __name__ == "__main__":
    run_dataframe_analysis()