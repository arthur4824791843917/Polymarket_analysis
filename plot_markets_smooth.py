import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.optimize import minimize

from data_loader import fetch_and_filter_markets as load_data


def fit_lognormal_cdf_for_bucket(strikes, cdf_vals, weights, grid, spot_price=None):
    sort_idx = np.argsort(strikes)
    strikes = np.array(strikes[sort_idx], dtype=float)
    cdf_vals = np.array(cdf_vals[sort_idx], dtype=float)
    weights = np.array(weights[sort_idx], dtype=float)

    total_w = np.sum(weights)
    weights = weights / total_w if total_w > 0 else np.ones_like(weights) / len(weights)

    init_scale = spot_price if (spot_price is not None and pd.notna(spot_price)) else np.median(strikes)
    init_s = 0.05

    def objective(params):
        s, scale = params
        if s <= 1e-4 or scale <= 0:
            return 1e9
        model_cdf = lognorm.cdf(strikes, s=s, scale=scale)
        return np.sum(weights * (model_cdf - cdf_vals) ** 2)

    res = minimize(
        objective,
        x0=[init_s, init_scale],
        bounds=[(0.001, 1.0), (grid[0] * 0.5, grid[-1] * 1.5)],
        method='L-BFGS-B'
    )

    fitted_s, fitted_scale = res.x
    
    # Calculate analytical mean of the fitted log-normal distribution: E[X] = scale * exp(s^2 / 2)
    expected_val = fitted_scale * np.exp((fitted_s ** 2) / 2.0)

    grid_cdf = lognorm.cdf(grid, s=fitted_s, scale=fitted_scale)
    return np.clip(grid_cdf, 0.0, 1.0), expected_val


def compute_rigorous_matrix(df, price_grid, date_grid, cutoff_date=None, stock_df=None):
    num_p = len(price_grid) - 1
    num_d = len(date_grid) - 1
    matrix = np.zeros((num_p, num_d))
    expected_prices = [np.nan] * num_d

    for d_idx in range(num_d):
        d_start = date_grid[d_idx]
        d_end = date_grid[d_idx + 1]

        if cutoff_date is not None and d_start <= cutoff_date:
            continue

        active = df[(df["Start Date"] <= d_end) & (df["Closing Date"] >= d_start)]
        if active.empty:
            continue

        cdf_points = []
        for _, row in active.iterrows():
            cond = row["Condition Type"]
            prob = row["Implied Odds (%)"] / 100.0
            liq = row["Pool Liquidity ($)"]

            if cond == "above" and pd.notna(row["Min Price Target"]):
                cdf_points.append((float(row["Min Price Target"]), 1.0 - prob, liq))
            elif cond == "below" and pd.notna(row["Max Price Target"]):
                cdf_points.append((float(row["Max Price Target"]), prob, liq))

        if len(cdf_points) >= 2:
            cdf_df = pd.DataFrame(cdf_points, columns=["price", "cdf_val", "weight"])
            grouped_rows = []
            for price_val, group in cdf_df.groupby("price"):
                w_sum = group["weight"].sum()
                avg_cdf = np.average(group["cdf_val"], weights=group["weight"]) if w_sum > 0 else group["cdf_val"].mean()
                grouped_rows.append({"price": price_val, "cdf_val": avg_cdf, "weight": w_sum})

            grouped = pd.DataFrame(grouped_rows).sort_values("price")

            spot_price = None
            if stock_df is not None and not stock_df.empty and "Close" in stock_df.columns:
                spot_price = stock_df["Close"].dropna().iloc[-1]

            # Fit Log-Normal and retrieve expected price value
            grid_cdf, exp_price = fit_lognormal_cdf_for_bucket(
                grouped["price"].values,
                grouped["cdf_val"].values,
                grouped["weight"].values,
                price_grid,
                spot_price=spot_price
            )

            pdf = np.diff(grid_cdf)
            matrix[:, d_idx] = pdf * 100.0
            expected_prices[d_idx] = exp_price

    return matrix, expected_prices


def plot_market_density(ticker="AAPL"):
    df = load_data()
    if df.empty:
        print("No market data found.")
        return

    df["Start Date"] = pd.to_datetime(df["Start Date"]).dt.tz_localize(None)
    df["Closing Date"] = pd.to_datetime(df["Closing Date"]).dt.tz_localize(None)

    min_date = df["Start Date"].min().normalize()
    max_date = df["Closing Date"].max().normalize() + pd.Timedelta(days=1)

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

    mesh = ax.pcolormesh(
        date_grid,
        price_grid,
        density_matrix,
        cmap="turbo",
        shading="flat",
        alpha=0.85,
        vmax=np.percentile(density_matrix[density_matrix > 0], 98) if np.any(density_matrix > 0) else None
    )
    fig.colorbar(mesh, ax=ax, label="Implied Probability Density (% per $1)")

    # 1. Historical Actual Stock Price Line
    if not stock_df.empty and "Close" in stock_df.columns:
        ax.plot(
            stock_df.index,
            stock_df["Close"],
            color="black",
            linewidth=2.5,
            label=f"{ticker} True Stock Price",
            zorder=10,
        )

        ax.axvline(
            cutoff_date,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label="Historical Cutoff",
            zorder=9,
        )

    # 2. Implied Expected Price Line (Forward Log-Normal Mean)
    # Using midpoint of date grid bins for alignment
    mid_dates = date_grid[:-1] + (date_grid[1:] - date_grid[:-1]) / 2
    ax.plot(
        mid_dates,
        expected_prices,
        color="cyan",
        linewidth=2.5,
        label="Expected Price Line E[P]",
        zorder=11,
    )

    ax.set_title(f"{ticker} Implied Probability Density & Expected Price Trajectory", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price ($)")
    ax.set_xlabel("Date")
    ax.set_ylim(280, 400)
    plt.xticks(rotation=45)
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_market_density("AAPL")