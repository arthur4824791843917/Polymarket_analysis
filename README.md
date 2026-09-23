# Polymarket AAPL Price Target Analysis Pipeline

A modular Python framework to programmatically query, filter, and analyze Apple (AAPL) prediction markets on Polymarket using a **Single Source of Truth** architecture.

## Repository Structure

* **`data_loader.py`**: Central hub that handles API fetching from Polymarket's Gamma/CLOB endpoints, filters out irrelevant market cap bets, and returns a clean, structured Pandas DataFrame.
* **`analyse_data.py`**: Performs macro-level market summaries, liquidity share calculations, and outputs top market rankings.
* **`plot_markets.py`**: Allows data to be intutively visualllised.


### 1. Install Dependencies
```bash
pip install -r requirements.txt