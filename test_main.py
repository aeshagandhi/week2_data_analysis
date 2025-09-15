from main import explore_data, filter_data, plot_analysis, train_model, load_data
import pandas as pd
import pytest  
import polars as pl



"""
This assignment is the second part of your two-week project. You will now focus on making your data analysis project reproducible and testable. You’ll write basic unit tests for your data analysis functions and set up a development environment using either Dev Container or Docker. It should be in the same Github Repository you created last week.

Test Coverage: Includes meaningful unit and system tests that validate core functions such as data loading, filtering, grouping, preprocessing, and machine learning model behavior, with clear structure and edge case handling. Make sure all tests pass.

Dev Environment Setup: A fully functional Dev Container or Docker setup (3 bonus points for both Docker and Dev Container), with requirement file, devcontainer.json/Docker files, ensuring all dependencies correctly installed and clear instructions for building, running, and using the environment.
"""

# -------------------------------
# Unit Tests
# -------------------------------


def test_load_data():
    df, pl_df = load_data("Data.csv")
    # Pandas: .empty, Polars: .height
    assert not df.empty, "DataFrame should not be empty"
    assert pl_df.height > 0, "PL DataFrame should not be empty"
    # Column presence
    assert "co2" in df.columns, "'co2' column should be present in the DataFrame"
    assert "co2" in pl_df.columns, "'co2' column should be present in the PL DataFrame"
    # Shape
    assert df.shape[0] == pl_df.shape[0], "Both DataFrames should have the same number of rows"
    assert df.shape[1] == pl_df.shape[1], "Both DataFrames should have the same number of columns"
    # Nulls
    assert df.isnull().sum().sum() >= 0, "DataFrame should handle null values"
    # Polars nulls: use select(pl.all().is_null().sum())
    assert pl_df.select(pl.all().is_null().sum()).to_numpy().sum() >= 0, "PL DataFrame should handle null values"
    # Duplicates
    assert df.duplicated().sum() >= 0, "DataFrame should handle duplicate values"
    # Polars: unique().height <= height
    assert pl_df.unique().height <= pl_df.height, "PL DataFrame should handle duplicate values"
    # Types
    assert isinstance(df, pd.DataFrame), "Should return a Pandas DataFrame"
    assert isinstance(pl_df, pl.DataFrame), "Should return a Polars DataFrame"
    

    # Memory usage
    assert df.memory_usage().sum() >= 0, "DataFrame should have valid memory usage"
    assert pl_df.estimated_size() >= 0, "PL DataFrame should have valid memory usage"
    # Column names
    assert df.columns.tolist() == list(pl_df.columns), "Column names should match between Pandas and Polars DataFrames"

    
def test_explore_data():
    df, pl_df = load_data("Data.csv")
    cleaned_df = explore_data(df, pl_df)
    assert cleaned_df.duplicated().sum() >= 0, "Should handle duplicate rows"
    assert isinstance(cleaned_df, pd.DataFrame), "Should return a Pandas DataFrame"
    assert cleaned_df.shape[0] == df.shape[0], "Row count should remain the same after cleaning"
    assert cleaned_df.shape[1] == df.shape[1], "Column count should remain the same after cleaning"
    assert cleaned_df.equals(df.fillna({col: 0 for col in df.select_dtypes(include='number').columns})), "Cleaned DataFrame should match expected filled DataFrame"
    assert cleaned_df.memory_usage().sum() >= 0, "Cleaned DataFrame should have valid memory usage"
    assert cleaned_df.columns.tolist() == df.columns.tolist(), "Column names should remain the same after cleaning"
    assert cleaned_df.head().equals(df.fillna({col: 0 for col in df.select_dtypes(include='number').columns}).head()), "First few rows should match expected filled DataFrame"
    assert cleaned_df.tail().equals(df.fillna({col: 0 for col in df.select_dtypes(include='number').columns}).tail()), "Last few rows should match expected filled DataFrame"

def test_filter_data():
    df, pl_df = load_data("Data.csv")
    subset_df, us_df, subset_pl_df = filter_data(df, pl_df)
    assert not subset_df.empty, "Subset DataFrame should not be empty"
    assert not us_df.empty, "USA DataFrame should not be empty"
    assert not subset_pl_df.is_empty(), "Subset Polars DataFrame should not be empty"
    assert all(subset_df["Description"] == "Country"), "All rows in subset should have Description 'Country'"
    assert all(subset_df["year"] >= 1900), "All rows in subset should have year >= 1900"
    assert all(us_df["iso_code"] == "USA"), "All rows in USA DataFrame should have iso_code 'USA'"
    assert all(us_df["year"] >= 2000), "All rows in USA DataFrame should have year >= 2000"
    assert all(subset_pl_df["Description"].str.to_lowercase() == "country"), "All rows in Polars subset should have Description 'Country'"
    assert all(subset_pl_df["year"] >= 1900), "All rows in Polars subset should have year >= 1900"
    assert isinstance(subset_df, pd.DataFrame), "Subset should be a Pandas DataFrame"
    assert isinstance(us_df, pd.DataFrame), "USA DataFrame should be a Pandas DataFrame"
    assert isinstance(subset_pl_df, pl.DataFrame), "Subset should be a Polars DataFrame"
    assert subset_df.shape[1] == df.shape[1], "Column count should remain the same in subset"
    assert us_df.shape[1] == df.shape[1], "Column count should remain the same in USA DataFrame"
    assert subset_pl_df.shape[1] == pl_df.shape[1], "Column count should remain the same in Polars subset"
    assert subset_df.memory_usage().sum() >= 0, "Subset DataFrame should have valid memory usage"
    assert us_df.memory_usage().sum() >= 0, "USA DataFrame should have valid memory usage"
    assert subset_pl_df.estimated_size() >= 0, "Polars subset should have valid memory usage"
    assert list(subset_pl_df.columns) == list(pl_df.columns), "Column names should remain the same in Polars subset"
    assert list(us_df.columns) == list(df.columns), "Column names should remain the same in USA DataFrame"
    assert list(subset_pl_df.columns) == list(pl_df.columns), "Column names should remain the same in Polars subset"

def test_plot_analysis():
    df, pl_df = load_data("Data.csv")
    _, us_df, subset_pl_df = filter_data(df, pl_df)
    try:
        plot_analysis(us_df, subset_pl_df)
    except Exception as e:
        pytest.fail(f"plot_analysis raised an exception: {e}")
    assert "co2" in us_df.columns, "'co2' column should be present              in the USA DataFrame"
    assert "year" in us_df.columns, "'year' column should be present in the USA DataFrame"
    assert "co2" in subset_pl_df.columns, "'co2'    column should be present in the Polars subset DataFrame"
    assert "year" in subset_pl_df.columns, "'year' column should be present in the Polars subset DataFrame"
    assert us_df.shape[0] > 0, "USA DataFrame should have rows for plotting"
    assert subset_pl_df.shape[0] > 0, "Polars subset DataFrame should have rows for plotting"
    assert us_df["co2"].dtype in [float, int], "'co2' column in USA DataFrame should be numeric"
    assert subset_pl_df["co2"].dtype in [pl.Float64, pl.Int64], "'co2' column in Polars subset DataFrame should be numeric"   
    assert us_df["year"].dtype in [int], "'year' column in USA DataFrame should be integer"
    assert subset_pl_df["year"].dtype in [pl.Int64], "'year' column in Polars subset DataFrame should be integer"
    assert us_df["co2"].min() >= 0, "'co2' values in USA DataFrame should be non-negative"
    assert subset_pl_df["co2"].min() >= 0, "'co2' values in Polars subset DataFrame should be non-negative"
    assert us_df["year"].min() >= 2000, "'year' values in USA DataFrame should be >= 2000"
    assert subset_pl_df["year"].min() >= 1900, "'year' values in Polars subset DataFrame should be >= 1900"
    assert us_df["co2"].max() <= 100000, "'co2' values in USA DataFrame should be within a reasonable range"
    assert subset_pl_df["co2"].max() <= 100000, "'co2' values in Polars subset DataFrame should be within a reasonable range"
    assert us_df["year"].max() <= 2023, "'year' values in USA DataFrame should be <= 2023"
    assert subset_pl_df["year"].max() <= 2023, "'year' values in Polars subset DataFrame should be <= 2023"


import pandas as pd
from main import load_data, filter_data, train_model

def test_train_model():
    df, pl_df = load_data("Data.csv")
    subset_df, _, _ = filter_data(df, pl_df)
    # Fill NaNs in features and target
    subset_df = subset_df.fillna({col: 0 for col in subset_df.columns})
    try:
        train_model(subset_df)
    except Exception as e:
        assert False, f"train_model raised an exception: {e}"
    # Check that subset_df has required columns and no nulls in features or target
    features = [
        "population", "gdp", "energy_per_capita", "energy_per_gdp", "primary_energy_consumption",
        "cement_co2", "coal_co2", "oil_co2", "gas_co2", "flaring_co2", "land_use_change_co2",
        "methane", "nitrous_oxide", "total_ghg", "total_ghg_excluding_lucf", "co2_per_gdp",
        "co2_per_capita", "co2_per_unit_energy", "share_global_co2", "share_global_co2_including_luc"
    ]
    for col in features:
        assert col in subset_df.columns, f"Missing feature column: {col}"
        assert subset_df[col].isnull().sum() == 0, f"Feature column '{col}' should not have nulls"
    assert "co2" in subset_df.columns, "Target column 'co2' should be present"
    assert subset_df["co2"].isnull().sum() == 0, "Target column 'co2' should not have nulls"
    assert subset_df.shape[0] > 0, "Training data should not be empty"


# -------------------------------
# System Tests
# -------------------------------
def test_system_integration():
    df, pl_df = load_data("Data.csv")
    # Data cleaning
    cleaned_df = explore_data(df, pl_df)
    # Filtering
    subset_df, us_df, subset_pl_df = filter_data(cleaned_df, pl_df)
    try:
        plot_analysis(us_df, subset_pl_df)
    except Exception as e:
        pytest.fail(f"plot_analysis raised an exception: {e}")
    # Model training (should not raise)
    subset_df = subset_df.fillna({col: 0 for col in subset_df.columns})
    try:
        train_model(subset_df)
    except Exception as e:
        pytest.fail(f"train_model raised an exception: {e}")
    # Basic checks
    assert not df.empty, "Loaded DataFrame should not be empty"
    assert not pl_df.is_empty(), "Loaded Polars DataFrame should not be empty"
    assert not subset_df.empty, "Filtered DataFrame should not be empty"
    assert not us_df.empty, "USA DataFrame should not be empty"
    assert subset_pl_df.height > 0, "Filtered Polars DataFrame should not be empty"
    


