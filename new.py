import pandas as pd
import numpy as np          # Added numpy
import scipy.stats as stats # Added scipy
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("businessemploymentdata1.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Drop rows with missing data
df = df.dropna(subset=['Data_value'])

# Convert Period to datetime if needed
df['Period'] = pd.to_datetime(df['Period'], errors='coerce')

# 1. Line Plot: Trend of Employment Data Over Time (overall)
plt.figure(figsize=(12, 6))
sns.lineplot(x='Period', y='Data_value', data=df)
plt.title('Employment Trend Over Time')
plt.xlabel('Period')
plt.ylabel('Employment Data Value')
plt.tight_layout()
plt.show()

# 2. Bar Plot: Top 10 Series with Highest Mean Data Value
top_series = df.groupby('Series_title_1')['Data_value'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_series.values, y=top_series.index, palette='viridis')
plt.title('Top 10 Series (by Series_title_1) with Highest Mean Data Value')
plt.xlabel('Mean Data Value')
plt.ylabel('Series Title 1')
plt.tight_layout()
plt.show()

# 3. Boxplot: Distribution of Data Value by Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='STATUS', y='Data_value', data=df)
plt.title('Distribution of Data Values by Status')
plt.tight_layout()
plt.show()

# 4. Count Plot: Frequency of Different Units Used
plt.figure(figsize=(10, 5))
sns.countplot(y='UNITS', data=df, order=df['UNITS'].value_counts().index)
plt.title('Count of Different Units')
plt.tight_layout()
plt.show()

# 5. Heatmap: Correlation Matrix of Numeric Columns
plt.figure(figsize=(8, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# 6. Pie Chart: Distribution of Subject Categories
subject_counts = df['Subject'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(subject_counts, labels=subject_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Distribution of Subject Categories")
plt.tight_layout()
plt.show()

# 7. Pair Plot of Top 5 Series
pivot_df = df.pivot_table(values='Data_value', index='Period', columns='Series_title_1', aggfunc='mean')
pivot_df = pivot_df.dropna(axis=1, thresh=int(0.7 * len(pivot_df)))
pivot_df = pivot_df.dropna()
selected_columns = pivot_df.columns[:5]
sns.pairplot(pivot_df[selected_columns])
plt.show()
