import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Plot
plt.figure(figsize=(8,6))

for cluster in range(5):
    data = df[df['Cluster'] == cluster]
    plt.scatter(
        data['Annual Income (k$)'],
        data['Spending Score (1-100)'],
        label=f'Cluster {cluster}'
    )

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=200,
    marker='X',
    label='Centroids'
)

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.legend()
plt.show()