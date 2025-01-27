import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

# Preview the datasets
print("Customers Data:")
print(customers.head())
print("\nProducts Data:")
print(products.head())
print("\nTransactions Data:")
print(transactions.head())

# Merge datasets
merged_data = transactions.merge(customers, on="CustomerID", how="left").merge(products, on="ProductID", how="left")

# Data Overview
print("\nMerged Data Overview:")
print(merged_data.info())
print(merged_data.describe())

# Checking for missing values
print("\nMissing Values:")
print(merged_data.isnull().sum())

# EDA
# 1. Distribution of Transactions by Region
plt.figure(figsize=(8, 5))
sns.countplot(data=merged_data, x="Region", order=merged_data['Region'].value_counts().index)
plt.title("Transactions by Region")
plt.xlabel("Region")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# 2. Top 10 Most Sold Products
top_products = merged_data.groupby("ProductName")["Quantity"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
top_products.plot(kind='bar', color='skyblue')
plt.title("Top 10 Most Sold Products")
plt.xlabel("Product Name")
plt.ylabel("Total Quantity Sold")
plt.xticks(rotation=45)
plt.show()

# 3. Revenue by Product Category
revenue_by_category = merged_data.groupby("Category")["TotalValue"].sum().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
revenue_by_category.plot(kind='bar', color='orange')
plt.title("Revenue by Product Category")
plt.xlabel("Category")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.show()

# 4. Time Series Analysis of Total Revenue
merged_data["TransactionDate"] = pd.to_datetime(merged_data["TransactionDate"])
revenue_by_date = merged_data.groupby("TransactionDate")["TotalValue"].sum()
plt.figure(figsize=(12, 6))
revenue_by_date.plot(color='green')
plt.title("Total Revenue Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.show()

# 5. Customer Lifetime Value (CLV) Analysis
clv = merged_data.groupby("CustomerID")["TotalValue"].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
clv.head(10).plot(kind='bar', color='purple')
plt.title("Top 10 Customers by Lifetime Value")
plt.xlabel("Customer ID")
plt.ylabel("Total Revenue")
plt.show()

# Save the merged data for further analysis
merged_data.to_csv("Merged_Data.csv", index=False)
