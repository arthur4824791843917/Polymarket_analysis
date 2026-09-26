import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.stats import norm
from scipy.optimize import minimize

from data_loader import fetch_and_filter_markets as load_data


def fit_normal_cdf_for_bucket(strikes, cdf_vals, weights, grid, spot_price=None):
    """Fits a Normal Distribution CDF (mean mu, std dev sigma) to discrete market odds."""
    sort_idx = np.argsort(strikes)
    strikes = np.array(strikes[sort_idx], dtype=float)
    cdf_vals = np.array(cdf_vals[sort_idx], dtype=float)
    weights = np.array(weights[sort_idx], dtype=float)

    total_w = np.sum(weights)
    weights = weights / total_w if total_w > 0 else np.ones_like(weights) / len(weights)

    # Initial guess: Mean at current spot price (or median strike), initial std dev = 15.0
    init_mu = spot_price if (spot_price is not None and pd.notna(spot_price)) else np.median(strikes)
    init_sigma = 15.0

    def objective(params):
        mu, sigma = params
        if sigma <= 1e-2:
            return 1e9
        model_cdf = norm.cdf(strikes, loc=mu, scale=sigma)
        return np.sum(weights * (model_cdf - cdf_vals) ** 2)

    res = minimize(
        objective,
        x0=[init_mu, init_sigma],
        bounds=[(grid[0], grid[-1]), (1.0, 100.0)],
        method='L-BFGS-B'
    )

    fitted_mu, fitted_sigma = res.x if res.success else (init_mu, init_sigma)

    # Generate exact Normal CDF across the price grid
    grid_cdf = norm.cdf(grid, loc=fitted_mu, scale=fitted_sigma)
    return np.clip(grid_cdf, 0.0, 1.0), fitted_mu


def compute_rigorous_matrix(df, price_grid, date_grid, cutoff_date=None, stock_df=None, use_log_scaling=False):
    num_p = len(price_grid) - 1
    num_d = len(date_grid) - 1
    raw_matrix = np.zeros((num_p, num_d))
    slice_liquidity = np.zeros(num_d)
    expected_prices = [np.nan] * num_d

    # Fetch latest spot price for initial optimization guess
    spot_price = None
    if stock_df is not None and not stock_df.empty and "Close" in stock_df.columns:
        spot_price = stock_df["Close"].dropna().iloc[-1]

    for d_idx in range(num_d):
        d_start = date_grid[d_idx]
        d_end = date_grid[d_idx + 1]

        if cutoff_date is not None and d_start <= cutoff_date:
            continue

        active = df[(df["Start Date"] <= d_end) & (df["Closing Date"] >= d_start)]
        if active.empty:
            continue

        total_liq = active["Pool Liquidity ($)"].sum()
        slice_liquidity[d_idx] = total_liq

        cdf_points = []
        for _, row in active.iterrows():
            cond = row["Condition Type"]
            prob = row["Implied Odds (%)"] / 100.0
            liq = max(0.0, float(row.get("Pool Liquidity ($)", 0.0)))

            if cond == "above" and pd.notna(row["Min Price Target"]):
                cdf_points.append((float(row["Min Price Target"]), 1.0 - prob, liq))
            elif cond == "below" and pd.notna(row["Max Price Target"]):
                cdf_points.append((float(row["Max Price Target"]), prob, liq))
            elif cond == "exact" and pd.notna(row["Min Price Target"]):
                cdf_points.append((float(row["Min Price Target"]), 1.0 - prob, liq))
            elif cond == "range":
                if pd.notna(row["Min Price Target"]):
                    cdf_points.append((float(row["Min Price Target"]), max(0.0, 0.5 - prob / 2.0), liq))
                if pd.notna(row["Max Price Target"]):
                    cdf_points.append((float(row["Max Price Target"]), min(1.0, 0.5 + prob / 2.0), liq))

        if len(cdf_points) >= 2:
            cdf_df = pd.DataFrame(cdf_points, columns=["price", "cdf_val", "weight"])
            grouped_rows = []

            for price_val, group in cdf_df.groupby("price"):
                w_sum = group["weight"].sum()
                weighted_cdf = (group["cdf_val"] * (group["weight"] / w_sum)).sum() if w_sum > 0 else group["cdf_val"].mean()
                grouped_rows.append({"price": price_val, "cdf_val": weighted_cdf, "weight": w_sum})

            grouped = pd.DataFrame(grouped_rows).sort_values("price")

            # Fit Normal CDF to produce a smooth bell-shaped PDF
            grid_cdf, mean_price = fit_normal_cdf_for_bucket(
                grouped["price"].values, grouped["cdf_val"].values, grouped["weight"].values, price_grid, spot_price=spot_price
            )

            pdf = np.diff(grid_cdf)
            raw_matrix[:, d_idx] = pdf * 100.0
            expected_prices[d_idx] = mean_price

    # Scale columns by pool liquidity ratio
    max_liq = np.max(slice_liquidity) if np.max(slice_liquidity) > 0 else 1.0
    liq_weights = (np.log1p(slice_liquidity) / np.log1p(max_liq)) if use_log_scaling else (slice_liquidity / max_liq)
    
    matrix = raw_matrix * liq_weights[np.newaxis, :]
    return matrix, expected_prices


def plot_market_density(ticker="AAPL"):
    df = load_data(query=ticker)
    if df.empty:
        print("No market data found.")
        return

    df["Start Date"] = pd.to_datetime(df["Start Date"], dayfirst=True, errors="coerce").dt.tz_localize(None)
    df["Closing Date"] = pd.to_datetime(df["Closing Date"], dayfirst=True, errors="coerce").dt.tz_localize(None)

    valid_starts = df["Start Date"].fillna(df["Closing Date"]).dropna()
    valid_closes = df["Closing Date"].dropna()

    if valid_starts.empty or valid_closes.empty:
        print("No valid date range in market data.")
        return

    min_date = valid_starts.min().normalize()
    max_date = valid_closes.max().normalize() + pd.Timedelta(days=1)

    price_grid = np.arange(280.0, 400.25, 0.25)
    date_grid = pd.date_range(start=min_date, end=max_date, freq="2h")

    try:
        stock_df = yf.download(ticker, start=min_date, end=max_date, progress=False)
        if isinstance(stock_df.columns, pd.MultiIndex):
            stock_df.columns = stock_df.columns.get_level_values(0)
    except Exception:
        stock_df = pd.DataFrame()

    cutoff_date = None
    if not stock_df.empty and "Close" in stock_df.columns:
        stock_df = stock_df.dropna(subset=["Close"])
        if not stock_df.empty:
            cutoff_date = stock_df.index.max().tz_localize(None)

    density_matrix, expected_prices = compute_rigorous_matrix(
        df, price_grid, date_grid, cutoff_date=cutoff_date, stock_df=stock_df
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    valid_vals = density_matrix[density_matrix > 0]
    max_val = np.percentile(valid_vals, 99) if len(valid_vals) > 0 else 1.0

    mesh = ax.pcolormesh(
        date_grid[:-1],
        price_grid[:-1],
        density_matrix,
        cmap="turbo",
        norm=PowerNorm(gamma=1, vmin=0, vmax=max_val),
        shading="gouraud",
        alpha=0.9,
    )

    fig.colorbar(mesh, ax=ax, label="Liquidity-Weighted Normal Probability Density")

    if not stock_df.empty and "Close" in stock_df.columns:
        ax.plot(
            stock_df.index,
            stock_df["Close"],
            color="white",
            linewidth=1,
            label=f"{ticker} True Stock Price",
            zorder=10,
        ) 

        ax.axvline(
            cutoff_date,
            color="white",
            linestyle=":",
            linewidth=1.2,
            alpha=0.7,
            label="Historical Cutoff",
            zorder=9,
        )

    # Plot central expected price trajectory line
    mid_dates = date_grid[:-1] + (date_grid[1:] - date_grid[:-1]) / 2
    ax.plot(
        mid_dates,
        expected_prices,
        color="white",
        linestyle="--",
        linewidth=1,
        label="Implied Mean E[P]",
        zorder=11,
    )

    ax.set_title(f"{ticker} Parametric Normal Implied Probability Heatmap", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price ($)")
    ax.set_xlabel("Date")
    ax.set_ylim(280, 400)
    plt.xticks(rotation=45)
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("example_output.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_market_density("AAPL")