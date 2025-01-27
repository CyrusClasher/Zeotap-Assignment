import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# Merge datasets
data = transactions.merge(customers, on='CustomerID').merge(products, on='ProductID')

# Feature engineering
customer_summary = data.groupby('CustomerID').agg({
    'TotalValue': 'sum',   # Total spending
    'TransactionID': 'count',  # Number of transactions
    'Quantity': 'sum',   # Total quantity purchased
    'Price': 'mean',  # Average product price
}).rename(columns={
    'TotalValue': 'TotalSpending',
    'TransactionID': 'TransactionCount',
    'Quantity': 'TotalQuantity',
    'Price': 'AvgPrice'
})

# Add region information from Customers.csv
customer_summary = customer_summary.merge(customers[['CustomerID', 'Region']], on='CustomerID')

# Encode categorical features (e.g., Region)
customer_summary = pd.get_dummies(customer_summary, columns=['Region'], drop_first=True)

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_summary.drop('CustomerID', axis=1))

# Clustering with K-Means
cluster_results = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    db_index = davies_bouldin_score(scaled_features, clusters)
    cluster_results.append({'k': k, 'DBIndex': db_index})

# Optimal number of clusters based on the lowest DB Index
optimal_k = min(cluster_results, key=lambda x: x['DBIndex'])['k']
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customer_summary['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize clusters
sns.pairplot(customer_summary, hue='Cluster', diag_kind='kde', palette='tab10')
plt.suptitle('Customer Clusters Visualization', y=1.02)
plt.show()

# Report clustering metrics
print("Clustering Results:")
print(f"Optimal number of clusters: {optimal_k}")
print("Davies-Bouldin Index for each k:")
for result in cluster_results:
    print(f"k={result['k']}, DBIndex={result['DBIndex']:.4f}")
