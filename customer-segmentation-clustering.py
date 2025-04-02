# 1. Simulates customer data by generating synthetic data
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

# 2. Standardizes data to normalize scales and improve clustering accuracy
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Generates linkage matrix
linkage_matrix = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 5))

# 4. Plots dendogram
dendrogram(linkage_matrix)
plt.title('Customer Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# 5. Cuts dendogram into three clusters
labels = fcluster(linkage_matrix, t=3, criterion='maxclust')
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='rainbow')
plt.title('Customer Segments')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()