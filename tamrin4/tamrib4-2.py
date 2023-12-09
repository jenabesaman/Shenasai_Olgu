import numpy as np
from scipy.spatial import distance

# نقاط مورد نظر
p1 = np.array([1, 2])
p2 = np.array([3, 6])

# فاصله اقلیدسی
euclidean_distance = np.linalg.norm(p1 - p2)
print(f"Euclidean distance: {euclidean_distance}")

# فاصله ماهالانوبیس
covariance_matrix = np.array([[4, 1], [1, 2]])
inverse_covariance = np.linalg.inv(covariance_matrix)
mahalanobis_distance = distance.mahalanobis(p1, p2, inverse_covariance)
print(f"Mahalanobis distance: {mahalanobis_distance}")
