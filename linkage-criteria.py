# Imports libraries
from scipy.spatial.distance import pdist, squareform
import numpy as np
from scipy.cluster.hierarchy import linkage

# Creates dataset and distance matrix
data = np.array([[1, 2], [2, 3], [5, 8]])
distance_matrix = squareform(pdist(data, metric='euclidean')) 

# Calculates Single linkage (minimum)
single_linkage_min = np.min(distance_matrix[0, 1:])
print("Single Linkage Distance between Cluster A (point 1) and Cluster B (point 2, 3):", single_linkage_min)

# Calculates Single linkage (maximum)
single_linkage_max = np.max(distance_matrix[0, 1:])
print("Complete Linkage Distance between Cluster A and B:", single_linkage_max)

# Calculates Average Linkage (Mean)
average_linkage = np.mean(distance_matrix[0, 1:])
print("Average Linkage Distance between Cluster A and B:", average_linkage)

# Calculates Linkage Ward
linkage_ward = linkage(data, method='ward')
print("Ward's Linkage Matrix:\n", linkage_ward)