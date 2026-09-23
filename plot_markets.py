import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from data_loader import fetch_and_filter_markets as load_data


def fit_cdf_for_bucket(strikes, cdf_vals, weights, grid):
    sort_idx = np.argsort(strikes)
    strikes = np.array(strikes[sort_idx], dtype=float)
    cdf_vals = np.array(cdf_vals[sort_idx], dtype=float)
    weights = np.array(weights[sort_idx], dtype=float)

    total_w = np.sum(weights)
    weights = weights / total_w if total_w > 0 else np.ones_like(weights) / len(weights)
    n_points = len(strikes)

    def objective(y):
        return np.sum(weights * (y - cdf_vals) ** 2)

    constraints = [{'type': 'ineq', 'fun': lambda y, i=i: y[i + 1] - y[i] - 1e-4} for i in range(n_points - 1)]
    bounds = [(0.001, 0.999) for _ in range(n_points)]
    x0 = np.maximum.accumulate(np.clip(cdf_vals, 0.001, 0.999))

    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    fitted_cdf_vals = res.x if res.success else x0

    ext_strikes = np.concatenate([[grid[0]], strikes, [grid[-1]]])
    ext_cdf_vals = np.concatenate([[0.0], fitted_cdf_vals, [1.0]])

    ext_strikes, unique_indices = np.unique(ext_strikes, return_index=True)
    ext_cdf_vals = ext_cdf_vals[unique_indices]

    pchip = PchipInterpolator(ext_strikes, ext_cdf_vals, extrapolate=False)
    grid_cdf = np.nan_to_num(pchip(grid), nan=0.0)
    grid_cdf = np.where(grid < ext_strikes[0], 0.0, grid_cdf)
    grid_cdf = np.where(grid > ext_strikes[-1], 1.0, grid_cdf)

    return np.clip(grid_cdf, 0.0, 1.0)


def compute_rigorous_matrix(df, price_grid, date_grid, cutoff_date=None):
    num_p = len(price_grid) - 1
    num_d = len(date_grid) - 1
    matrix = np.zeros((num_p, num_d))

    for d_idx in range(num_d):
        d_start = date_grid[d_idx]
        d_end = date_grid[d_idx + 1]

        # Mask matrix out for dates prior to or on the cutoff date
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
            grid_cdf = fit_cdf_for_bucket(
                grouped["price"].values, grouped["cdf_val"].values, grouped["weight"].values, price_grid
            )

            # Compute raw PDF from CDF derivative
            pdf = np.diff(grid_cdf)
            
            # Smooth out spline steps using a 2-bin Gaussian filter
            smoothed_pdf = gaussian_filter1d(pdf, sigma=2.0)
            
            matrix[:, d_idx] = smoothed_pdf * 100.0

    return matrix


def plot_market_density(ticker="AAPL"):
    df = load_data()
    if df.empty:
        print("No market data found.")
        return

    df["Start Date"] = pd.to_datetime(df["Start Date"]).dt.tz_localize(None)
    df["Closing Date"] = pd.to_datetime(df["Closing Date"]).dt.tz_localize(None)

    min_date = df["Start Date"].min().normalize()
    max_date = df["Closing Date"].max().normalize() + pd.Timedelta(days=1)

    # Refined grid resolution: $0.25 price increments and 2-hour date sampling
    price_grid = np.arange(280.0, 400.25, 0.25)
    date_grid = pd.date_range(start=min_date, end=max_date, freq="2h")

    # Retrieve historical stock data
    try:
        stock_df = yf.download(ticker, start=min_date, end=max_date, progress=False)
        if isinstance(stock_df.columns, pd.MultiIndex):
            stock_df.columns = stock_df.columns.get_level_values(0)
    except Exception:
        stock_df = pd.DataFrame()

    # Determine last available trading date from stock data
    cutoff_date = None
    if not stock_df.empty and "Close" in stock_df.columns:
        stock_df = stock_df.dropna(subset=["Close"])
        if not stock_df.empty:
            cutoff_date = stock_df.index.max().tz_localize(None)

    # Calculate probability matrix starting after the cutoff date
    density_matrix = compute_rigorous_matrix(df, price_grid, date_grid, cutoff_date=cutoff_date)

    fig, ax = plt.subplots(figsize=(14, 8))

    mesh = ax.pcolormesh(
        date_grid,
        price_grid,
        density_matrix,
        cmap="turbo",
        shading="flat",
        alpha=0.85,
        vmax=np.percentile(density_matrix[density_matrix > 0], 95) if np.any(density_matrix > 0) else None
    )
    fig.colorbar(mesh, ax=ax, label="Implied Probability Density (% per $1)")

    # Plot real stock price up to the latest available point
    if not stock_df.empty and "Close" in stock_df.columns:
        ax.plot(
            stock_df.index,
            stock_df["Close"],
            color="black",
            linewidth=2.5,
            label=f"{ticker} True Stock Price",
            zorder=10,
        )

        # Vertical line indicating where market historical price ends and prediction matrix begins
        ax.axvline(
            cutoff_date,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label="Historical Cutoff",
            zorder=9,
        )

    ax.set_title(f"{ticker} Liquidity-Weighted Implied Probability Heatmap", fontsize=14, fontweight="bold")
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