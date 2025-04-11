import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Data Cleaning

df = pd.read_csv("real_data.csv")

# Remove trailing empty rows
df.iloc[18924:] = df.iloc[18924:].dropna(how='all')
df.to_csv("real_data_clean.csv", index=False)

# Reload cleaned dataset and handle missing values
df = pd.read_csv("real_data_clean.csv")
df = df.fillna(0)
df.to_csv("real_data1.csv", index=False)

# Load the final cleaned dataset
df = pd.read_csv("real_data1.csv")

# Convert release_date to datetime and extract release_year
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year

# Set plot style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# Regional Sales by Genre

region_sales_by_genre = df.groupby("genre")[["na_sales", "jp_sales", "pal_sales", "other_sales"]].sum().sort_values("na_sales", ascending=False)
region_sales_by_genre.plot(kind="bar", stacked=True)
plt.title("Regional Sales by Genre")
plt.ylabel("Sales (in millions)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Regional Sales by Console

region_sales_by_console = df.groupby("console")[["na_sales", "jp_sales", "pal_sales", "other_sales"]].sum().sort_values("na_sales", ascending=False).head(10)
region_sales_by_console.plot(kind="bar", stacked=True)
plt.title("Top 10 Consoles by Regional Sales")
plt.ylabel("Sales (in millions)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Critic Score vs Total Sales

sns.scatterplot(data=df, x="critic_score", y="total_sales", hue="genre", alpha=0.6)
plt.title("Critic Score vs Total Sales")
plt.xlabel("Critic Score")
plt.ylabel("Total Sales (millions)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#  Correlation Heatmap

correlation = df[["critic_score", "total_sales", "na_sales", "jp_sales", "pal_sales", "other_sales"]].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

#  Genre Popularity Over Time

genre_year_sales = df.groupby(["release_year", "genre"])["total_sales"].sum().reset_index()
plt.figure(figsize=(14, 7))
sns.lineplot(data=genre_year_sales, x="release_year", y="total_sales", hue="genre")
plt.title("Genre Popularity Over Time")
plt.ylabel("Total Sales (millions)")
plt.xlabel("Release Year")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Top Publishers by Sales 

top_publishers = df.groupby("publisher")["total_sales"].sum().sort_values(ascending=False).head(10)
top_publishers_df = top_publishers.reset_index()
top_publishers_df.columns = ["publisher", "total_sales"]

sns.barplot(data=top_publishers_df, x="total_sales", y="publisher", hue="publisher", palette="viridis", legend=False)
plt.title("Top 10 Publishers by Total Sales")
plt.xlabel("Total Sales (millions)")
plt.tight_layout()
plt.show()

# Top Developers by Critic Score 

top_developers = df.groupby("developer")["critic_score"].mean().sort_values(ascending=False).head(10)
top_developers_df = top_developers.reset_index()
top_developers_df.columns = ["developer", "critic_score"]

sns.barplot(data=top_developers_df, x="critic_score", y="developer", hue="developer", palette="magma", legend=False)
plt.title("Top 10 Developers by Average Critic Score")
plt.xlabel("Average Critic Score")
plt.tight_layout()
plt.show()


#Outlier Detection (Boxplots)

numeric_columns = ["critic_score", "total_sales", "na_sales", "jp_sales", "pal_sales", "other_sales"]

plt.figure(figsize=(14, 8))
sns.boxplot(data=df[numeric_columns], palette="Set2")
plt.title("Boxplot for Outlier Detection in Sales & Scores")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()





