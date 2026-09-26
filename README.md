# Polymarket Future Stock Price Analysis

Gambling informed by others gambling. 

A quantitative Python framework to pull prediction market odds from Polymarket API, derive smooth risk-neutral probability density functions (PDF), and visualize forward-looking price heatmaps alongside historical stock performance.

---

## Features

- **Live Polymarket Ingestion**: Directly queries Polymarket's public API to retrieve live condition targets, prices, and pool liquidities.
- **Natural Language Condition Parsing**: Regex-based extraction of target price bounds (`>` above, `<` below, `$X-$Y` range, and exact strike prices).
- **Parametric (Normal Distribution)**: Fits a Normal CDF $(\mu, \sigma)$ via `scipy.optimize.minimize` (L-BFGS-B) to construct central bell-curve probability density functions.
- **Liquidity-Weighted Density Mapping**: Scales probability intensity across time and price slices based on market pool liquidity ratios.
- **Historical Stock Overlay**: Integrates `yfinance` to automatically overlay actual stock price histories up to the historical cutoff date.

---

## Repository Structure

```text
.
├── data_loader.py          # Fetches, parses, and normalizes Polymarket prediction market data
├── analyse_data.py         # Summarizes active market liquidity, volume metrics, and top pools
├── plot_markets_smooth.py  # Parametric Normal CDF fitting and heatmap visualization
├── plot_markets.py         # Non-parametric PCHIP interpolation heatmap visualization
├── requirements.txt        # Python dependency specifications
└── README.md               # Project documentation