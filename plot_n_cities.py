import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# You may want more colors if you have many cities
COLORS = [
    "red",
    "blue",
    "green",
    "purple",
    "brown",
    "orange",
    "gray",
    "pink",
    "olive",
    "cyan",
]
plt.rcParams.update({"font.size": 14})


def plot_runs(
    folder_path: str,
    ax,
    mean: bool = True,
    n: float = 1.96,
    alpha: float = 1.0,
    name: str = None,
    color_offset: int = 0,
):
    """
    Plots the infected counts for multiple runs in a given folder, for an arbitrary number of cities.
    If mean=True, plots the average with fill between for confidence intervals; otherwise plots individual lines.
    """

    dataframes = []
    files = os.listdir(folder_path)
    files = [i for i in files if i.endswith(".csv") and i.startswith("SIR")]
    files = sorted(files)

    print(f"Processing files in {folder_path}: {len(files)}")

    for file in files:
        fpath = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(fpath)
            df = df.astype(float)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading file {fpath}: {e}")
            continue

    if not dataframes:
        print(f"No CSV files found in {folder_path}.")
        return

    # Identify columns that indicate infection counts
    # e.g. columns named "Infected - CityName"
    first_df = dataframes[0]
    infected_cols = [col for col in first_df.columns if "Infected - " in col]

    # Extract city names based on column headers
    # e.g. column = "Infected - Mumbai" => city = "Mumbai"
    cities = list(set([col.split("Infected - ")[-1].strip() for col in infected_cols]))

    # Prepare storage for all city data across runs
    data = {city: [] for city in cities}

    # Days (assuming all dataframes have the same 'Day' column)
    days = first_df["Day"]

    # Collect data for each city across all runs
    for i, df in enumerate(dataframes):
        for city in cities:
            data[city].append(df[f"Students - Infected - {city}"] + df[f"Adults - Infected - {city}"])

        # Plot individual runs if mean=False
        if not mean:
            for idx, city in enumerate(cities):
                days_df = df["Day"]
                inf = df[f"Students - Infected - {city}"] + df[f"Adults - Infected - {city}"]
                ax.plot(
                    days_df,
                    inf,
                    color=COLORS[
                        (idx + color_offset) % len(COLORS)
                    ],  # cycle through COLORS
                    alpha=alpha * 0.3,
                    linewidth=1,
                    label=f"Infected - {city}" if i == 0 else None,
                )

    # Plot mean and confidence intervals if mean=True
    if mean:
        for idx, city in enumerate(cities):
            city_arrays = np.array(data[city])  # shape: (num_runs, num_days)
            mean_data = np.mean(city_arrays, axis=0)
            std_dev = np.std(city_arrays, axis=0)

            ax.plot(
                days,
                mean_data,
                label=city,
                color=COLORS[(idx + color_offset) % len(COLORS)],
                alpha=alpha,
                linewidth=1,
            )

            ax.fill_between(
                days,
                mean_data - n * std_dev,
                mean_data + n * std_dev,
                color=COLORS[(idx + color_offset) % len(COLORS)],
                alpha=0.2 * alpha,
            )

    # Get parameters from folder name for title
    title = folder_path
    title = title + f" (Avg: #{len(dataframes)} runs)"

    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel("Number of Infected")
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    base_dir = "outputs"

    # Get all run directories
    run_dirs = ["test"]
    run_dirs = sorted(run_dirs)
    n_runs = len(run_dirs)

    print(f"Found {n_runs} runs in {base_dir}")

    # Calculate grid dimensions
    n_cols = int(np.ceil(np.sqrt(n_runs)))
    n_rows = int(np.ceil(n_runs / n_cols))

    # Create figure
    fig_size = (8 * n_cols + 3, 5 * n_rows + 3)
    if n_cols == 1 and n_rows == 1:
        fig_size = (15, 9)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    if n_runs == 1:
        # axes won't be an array if there's only one plot
        axes = np.array([axes])

    axes = axes.flatten()

    # Plot each run
    for idx, run_dir in enumerate(run_dirs):
        folder_path = os.path.join(base_dir, run_dir)
        plot_runs(
            folder_path,
            axes[idx],
            mean=False,
            name="Base Scenario",
            n=1,  # 1 => ~68% CI; for ~95% CI, set to 1.96
            alpha=0.8,
            # color_offset=3 * idx,
        )

    # Remove extra subplots
    for idx in range(n_runs, len(axes)):
        fig.delaxes(axes[idx])

    # Global title
    plt.suptitle("Infected Individuals Over Time (N Cities)", y=1.02)
    plt.tight_layout()
    plt.savefig("n_cities_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
