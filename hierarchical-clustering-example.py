# Imports libraries
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generates sample data and scales data
data = np.array([[1, 2], [2, 3], [5, 8], [8, 8], [1, 0], [0, 1]])
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Generates linkage matrix using Ward’s method
linkage_matrix = linkage(data_scaled, method='ward')
plt.figure(figsize=(8, 4))
plt.show()


# Plots dendogram
dendrogram(linkage_matrix)
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

# Cuts dendogram to get a specific number of clusters
clusters = fcluster(linkage_matrix, t=2, criterion='maxclust')
print(clusters)