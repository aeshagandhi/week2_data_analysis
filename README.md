# Week 2: Mini Assignment
Data analysis using Pandas/Polars on a Kaggle dataset
[![Python Template for IDS706](https://github.com/aeshagandhi/week2_data_analysis/actions/workflows/main.yml/badge.svg)](https://github.com/aeshagandhi/week2_data_analysis/actions/workflows/main.yml)

All exploratory data analysis code is contained in the main.py file. The original code was done in main.ipynb to use jupyter notebook for more interactive exploration, and then main.ipynb was converted into main.py for the python script submission.

About the data: The dataset is called CO₂ Emissions Across Countries, Regions, & Sectors, which includes detailed historical information on population, GDP, energy use, and emissions from cement, coal, oil, gas, flaring, and land-use change. The data is sourced from Our World in Data, but obtained from Kaggle. The dataset can be publicly found through Kaggle: https://www.kaggle.com/datasets/shreyanshdangi/co-emissions-across-countries-regions-and-sectors.

This project explores global carbon dioxide and greenhouse gas emissions data and focuses on patterns on countries and years. To complete the analysis, both Pandas and Polars were used for performance comparison and to do basic data cleaning/visualization. To create a machine learning model, a Random Forest Regressor from Scikit-learn was used to recognize patterns in beneficial features for predicting carbon dioxide emissions. 

Steps include:
1. Import the dataset by loading the Kaggle dataset and inspecting the data via .head(), .info(), and .describe(). The dataset is 13.77 MB and contains 43746 rows and 80 columns. While the intitial size is larger than 5 MB, filtering was done later to create a subset of the data.
2. Data cleaning included checking for missing values and duplicates and filling NaN values in mumeric columns with zeros.
3. Filtering and grouping: The dataset was filtered to only consider countries, i.e. not territories or other regions, and only consider years after 1900. The dataset was grouped by year to examine trends in carbon dioxide emissions over time and also grouped by country/year to consider mean gdp and population and mean co2.
4. Random Forest Machine Learning model: I Trained a Random Forest Regressor with 100 trees and split into 80/20 training-testing sets, with the target variable as total CO2 emissions. I evaluated the model using Mean Squared Error and R^2 score. Two models were created, the first one only had a few features ("population", "gdp", "cement_co2", "co2_per_capita") and then the second model inputted many more features such as population, gdp, primary_energy_consumption, cement_co2, coal_co2, oil_co2, methane, etc. The second model decreased the mean square error from 370 to 78.
5. Plotting: Feature Importance plot was used to interpret the results of the random forest and identify the strongest predictors of CO2 emissions, which included greenhouse gas totals and primary energy consumption. From the analysis, these two features can be suggested as the main drivers of carbon dixoide emissions.
6. Using Polars to explore the data and visualize the trends among the top 5 countries by mean CO2 emissions across all years since 1900. The stacked line plot shows that China has had rapid increase in emissions since 1990, while the U.S has been slowly decreasing their emissions in the last 20 years. Germany, Japan, and Russia have similar trends with each other while the U.S and China stand out more. 

Conclusion: Energy consumption and total greenhouse gas emissions are strong predictors of CO2 emissions, which seems reasonable as carbon dioxide and fossil fuels use are often connected. Some features such as population and GDP aren't as important as expected when considering the emissions, maybe because there isn't as direct of a relationship in comparison to the top two features mentioned. The second random forest achieved a reasonable R squared score, suggesting the model could be capturing the relationship between the emissions and energy use decently well. However, more analysis could be done at more specific granularity such as at the country level, since the model could be biased by countries that are large emitters of energy such as China, India, and US. I also visualized, via line plot, carbon dioxide emissions for the U.S over time from 2000 to 2024 and was surprised to see a general decrease in emissions, which could be potentially explained by a general shift in seeking alternative energy sources.


This project also includes the following development environment:
Dev Containers: The project includes a VS Code Dev Container for a consistent, reproducible development setup across machines.

Makefile: Common tasks (installing dependencies, formatting, linting, testing) are automated through a simple Makefile. For example:

make install   # install dependencies
make format    # auto-format with Black
make lint      # lint with flake8
make test      # run tests


GitHub Actions Workflow: A CI workflow (.github/workflows/python.yml) automatically runs on every push and pull request. It installs dependencies, lints the code with flake8, and ensures the codebase stays clean.