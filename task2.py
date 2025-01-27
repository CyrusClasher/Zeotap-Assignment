import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# Preprocessing Customers Data
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])

# Preprocessing Transactions Data
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

# Aggregate transaction data to create customer profiles
customer_profile = transactions.groupby('CustomerID').agg(
    TotalSpend=('TotalValue', 'sum'),
    AvgTransactionValue=('TotalValue', 'mean'),
    TotalTransactions=('TransactionID', 'count'),
    MostPurchasedCategory=('ProductID', lambda x: products.loc[products['ProductID'].isin(x)].Category.mode()[0])
).reset_index()

# Merge customer data with customer profiles
customer_data = pd.merge(customers, customer_profile, on='CustomerID', how='left')

# Encode categorical features
encoder = LabelEncoder()
customer_data['Region'] = encoder.fit_transform(customer_data['Region'])
customer_data['MostPurchasedCategory'] = encoder.fit_transform(customer_data['MostPurchasedCategory'].fillna('Unknown'))

# Fill missing values in numerical columns
customer_data.fillna(0, inplace=True)

# Select relevant features for similarity calculation
features = ['Region', 'TotalSpend', 'AvgTransactionValue', 'TotalTransactions', 'MostPurchasedCategory']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_data[features])

# Compute similarity matrix
similarity_matrix = cosine_similarity(scaled_features)

# Generate Lookalike recommendations
lookalike_results = {}
customer_ids = customer_data['CustomerID']

for idx, customer_id in enumerate(customer_ids[:20]):  # First 20 customers (C0001 to C0020)
    similarities = list(enumerate(similarity_matrix[idx]))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)  # Sort by similarity score
    top_3 = [(customer_ids[i], score) for i, score in similarities[1:4]]  # Exclude the customer itself
    lookalike_results[customer_id] = top_3

# Save Lookalike results to CSV
lookalike_df = pd.DataFrame({
    'CustomerID': list(lookalike_results.keys()),
    'Lookalikes': [str(value) for value in lookalike_results.values()]
})
lookalike_df.to_csv('Lookalike.csv', index=False)

print("Lookalike recommendations saved to Lookalike.csv")
