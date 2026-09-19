# process_data.py
import matplotlib.pyplot as plt
from data_loader import fetch_and_filter_markets, fetch_orderbook_liquidity

def display_market_tree_and_plot():
    df = fetch_and_filter_markets(query="AAPL", min_liquidity=1000.0)

    if df.empty:
        print("No active markets to display.")
        return

    # Print market details with orderbook depth
    for idx, row in df.iterrows():
        print(f"\n📌 Bet: {row['Market Question']}")
        print(f"   • Pool Liquidity:   ${row['Pool Liquidity ($)']:,.2f}")
        print(f"   • 24h Volume:       ${row['24h Volume ($)']:,.2f}")
        print(f"   • Yes Price/Odds:   ${row['Yes Price ($)']:.3f} ({row['Implied Odds (%)']}%)")

        if row["Tokens"]:
            ob_depth = fetch_orderbook_liquidity(row["Tokens"][0])
            print(f"     └ Active Orderbook Depth: ${ob_depth:,.2f}")

    # Plot histogram directly from DataFrame series
    plt.figure(figsize=(10, 6))
    plt.hist(df["Pool Liquidity ($)"], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Frequency Distribution of AAPL Market Liquidity (≥ $3,000)', fontsize=14)
    plt.xlabel('Pool Liquidity ($ USD)', fontsize=12)
    plt.ylabel('Frequency (Number of Markets)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    display_market_tree_and_plot()