import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# -------------------------------
# Data Loading
# -------------------------------
def load_data(filepath: str):
    """Load dataset into Pandas and Polars DataFrames."""
    df = pd.read_csv(filepath)
    pl_df = pl.read_csv(filepath)
    return df, pl_df


# -------------------------------
# Exploratory Data Analysis
# -------------------------------
def explore_data(df: pd.DataFrame, pl_df: pl.DataFrame):
    """Perform basic data inspection and cleaning."""
    print(df.head())
    print(pl_df.head())
    print("Shape:", df.shape)
    print(df.info())
    print(df.describe())
    print("Schema:", pl_df.schema)

    # Nulls and duplicates
    print("Null values per column:\n", df.isnull().sum())
    df = df.fillna({col: 0 for col in df.select_dtypes(
        include="number").columns})
    print("Duplicate rows:", df.duplicated().sum())

    return df


# -------------------------------
# Filtering and Grouping
# -------------------------------
def filter_data(df: pd.DataFrame, pl_df: pl.DataFrame):
    """Subset country-level data after 1900 in Pandas and Polars."""
    subset_df = df[(df["Description"] == "Country") & (df["year"] >= 1900)]

    # USA 21st century
    us_df = subset_df[(subset_df["iso_code"] == "USA") & (
        subset_df["year"] >= 2000)]

    # Polars filtering
    numeric_cols = [
        col for col, dtype in pl_df.schema.items() if dtype in [
            pl.Int64, pl.Float64]
    ]
    pl_df = pl_df.with_columns(
        [pl.col(col).fill_null(0) for col in numeric_cols])
    subset_pl_df = pl_df.filter(
        (pl.col("Description").str.to_lowercase() == "country")
        & (pl.col("year") >= 1900)
    )

    return subset_df, us_df, subset_pl_df


# -------------------------------
# Plotting and Analysis
# -------------------------------
def plot_analysis(us_df: pd.DataFrame, subset_pl_df: pl.DataFrame):
    """Create plots for exploratory analysis."""
    # US emissions over time
    plt.figure(figsize=(10, 6))
    plt.plot(us_df["year"], us_df["co2"], marker="o", linestyle="-")
    plt.title("U.S. CO₂ Emissions Over Time")
    plt.xlabel("Year")
    plt.ylabel("CO₂ Emissions (million tonnes)")
    plt.grid(True)
    plt.show()

    # Top 5 countries by mean CO2
    top_countries_df = (
        subset_pl_df.group_by("Name")
        .agg(pl.col("co2").mean().alias("mean_co2"))
        .sort("mean_co2", descending=True)
        .head(5)
    )
    top_countries = top_countries_df["Name"].to_list()

    # Filter and plot
    plot_df = subset_pl_df.filter(pl.col("Name").is_in(top_countries))
    plot_pd = plot_df.select(["year", "Name", "co2"]).to_pandas()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_pd, x="year", y="co2", hue="Name", marker="o")
    plt.title("CO₂ Emissions Over Time for Top 5 Countries")
    plt.xlabel("Year")
    plt.ylabel("CO₂ Emissions")
    plt.legend(title="Country")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -------------------------------
# Machine Learning Model
# -------------------------------
def train_model(subset_df: pd.DataFrame):
    """Train a Random Forest model to predict CO₂ emissions."""
    features = [
        "population",
        "gdp",
        "energy_per_capita",
        "energy_per_gdp",
        "primary_energy_consumption",
        "cement_co2",
        "coal_co2",
        "oil_co2",
        "gas_co2",
        "flaring_co2",
        "land_use_change_co2",
        "methane",
        "nitrous_oxide",
        "total_ghg",
        "total_ghg_excluding_lucf",
        "co2_per_gdp",
        "co2_per_capita",
        "co2_per_unit_energy",
        "share_global_co2",
        "share_global_co2_including_luc",
    ]

    X = subset_df[features].fillna(0)
    y = subset_df["co2"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

    # Feature importance plot
    importance = pd.Series(
        rf.feature_importances_, index=features).sort_values(
        ascending=False
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance.values, y=importance.index, palette="viridis")
    plt.title("Feature Importances for Predicting CO₂ Emissions")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


# -------------------------------
# Main Function
# -------------------------------
def main():
    filepath = "Data.csv"
    df, pl_df = load_data(filepath)

    df = explore_data(df, pl_df)
    subset_df, us_df, subset_pl_df = filter_data(df, pl_df)

    plot_analysis(us_df, subset_pl_df)
    train_model(subset_df)


if __name__ == "__main__":
    main()
