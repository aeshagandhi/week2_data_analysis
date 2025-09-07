#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import polars as pl
# pip install -r requirements.txt
# pip install jupyter

# pip install ipykernel jupyter
# python -m ipykernel install --user --name=myproject-venv --display-name "Python (myproject-venv)"

# python3 -m venv ~/.WEEK2_DATA_ANALYSIS
# source ~/.WEEK2_DATA_ANALYSIS/bin/activate


# ### Import the Dataset ###

# In[2]:


df = pd.read_csv('Data.csv')
pl_df = pl.read_csv('Data.csv')
# sourced from Kaggle: https://www.kaggle.com/datasets/shreyanshdangi/co-emissions-across-countries-regions-and-sectors


# ### Inspect the Data ###

# In[3]:


df.head()


# In[4]:


pl_df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


pl_df.schema


# In[9]:


# view null values per column
print(df.isnull().sum())


# In[10]:


df = df.fillna({col: 0 for col in df.select_dtypes(include="number").columns})


# In[11]:


print("Duplicate rows:", df.duplicated().sum())


# In[12]:


df


# ### Basic Filtering and Grouping ###

# Due to larger data, filter to only consider countries and years after 1900

# In[13]:


subset_df = df[(df['Description'] == 'Country')]
subset_df = subset_df[subset_df['year'] >= 1900]
subset_df.head()


# Look at only USA in the 21st century

# In[14]:


# look at the United States only considering 21st century 
us_df = subset_df[(subset_df["iso_code"] == "USA") & (subset_df["year"] >= 2000)]
us_df.head()


# In[15]:


import matplotlib.pyplot as plt


# line plot of total CO2 emissions over time
plt.figure(figsize=(10,6))
plt.plot(us_df['year'], us_df['co2'], marker='o', linestyle='-')
plt.title('U.S. CO₂ Emissions Over Time')
plt.xlabel('Year')
plt.ylabel('CO₂ Emissions (million tonnes)')
plt.grid(True)
plt.show()


# Look at mean CO2 emissions by country

# In[16]:


# group by country and take mean co2 
country_by_co2 = subset_df.groupby("Name")["co2"].mean().reset_index()
country_by_co2 


# In[17]:


# population and mean gdp per country and year
country_year_stats = subset_df.groupby(["Name", "year"])[["population", "gdp"]].mean().reset_index()
country_year_stats


# In[18]:


# global yearly co2 across countries
subset_df.groupby("year")["co2"].agg(["sum", "mean", "count"]).reset_index()


# ### Polars data cleaning and filtering ###
# 

# In[ ]:


# only include numeric cols
numeric_cols = [col for col, dtype in pl_df.schema.items() if dtype in [pl.Int64, pl.Float64]]

# fill nulls with 0
pl_df = pl_df.with_columns([
    pl.col(col).fill_null(0) for col in numeric_cols
])
subset_pl_df = pl_df.filter(
    (pl.col("Description").str.to_lowercase() == "country") & 
    (pl.col("year") >= 1900)
)


# In[ ]:


# top 5 countries by mean CO2
top_countries_df = (
    subset_pl_df
    .group_by("Name")
    .agg(pl.col("co2").mean().alias("mean_co2"))
    .sort("mean_co2", descending=True)
    .head(5)
)

top_countries = top_countries_df["Name"].to_list()

# filter data for these countries
plot_df = subset_pl_df.filter(pl.col("Name").is_in(top_countries))

# convert to Pandas for plotting
plot_pd = plot_df.select(["year", "Name", "co2"]).to_pandas()

# plot
import seaborn as sns

plt.figure(figsize=(10,6))
sns.lineplot(data=plot_pd, x="year", y="co2", hue="Name", marker="o")
plt.title("CO₂ Emissions Over Time for Top 5 Countries")
plt.xlabel("Year")
plt.ylabel("CO₂ Emissions")
plt.legend(title="Country")
plt.grid(True)
plt.tight_layout()
plt.show()


# ### Explore a Machine Learning Algorithm ###

# First try some basic features such as 

# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# pick features
features = ["population", "gdp", "cement_co2", "co2_per_capita"]

X = subset_df[features]
y = subset_df["co2"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# intiialize rf
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# train and predict
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# evaluate
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))



# Try some more features

# In[22]:


# now add more features for the random forest to subsample from
features = [
    'population', 'gdp', 'energy_per_capita', 'energy_per_gdp',
    'primary_energy_consumption', 'cement_co2', 'coal_co2', 'oil_co2',
    'gas_co2', 'flaring_co2', 'land_use_change_co2',
    'methane', 'nitrous_oxide', 'total_ghg', 'total_ghg_excluding_lucf',
    'co2_per_gdp', 'co2_per_capita', 'co2_per_unit_energy',
    'share_global_co2', 'share_global_co2_including_luc'
]
X = subset_df[features]
y = subset_df["co2"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# intiialize rf
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# train and predict
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# evaluate
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


# 

# 

# Look at feature importances

# In[23]:


importance = pd.Series(rf.feature_importances_, index=features)
print(importance.sort_values(ascending=False))


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns

top_features = importance.sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title("Feature Importances for Predicting CO₂ Emissions")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# In[ ]:




