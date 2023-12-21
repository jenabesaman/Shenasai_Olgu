import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

x = np.array([[-2, -1], [-2, -2], [-1, 0], [-1, -1], [-1, -3], [0, -1], [1, -3], [2, -1]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

lda = LinearDiscriminantAnalysis(n_components=1)
x_lda = lda.fit_transform(x, y)

# The coefficients of the separating line
print('Coefficients of the separating line: ', lda.coef_)

import matplotlib.pyplot as plt

# Plot the original data points
plt.figure(figsize=(8, 6))
plt.scatter(x[:, 0], x[:, 1], c=y)

# Create a range of values for the x-axis
x_values = np.linspace(-3, 3, 400)

# Calculate the corresponding y values for the separating line
y_values = -(lda.coef_[0][0] * x_values) / lda.coef_[0][1]

# Plot the separating line
plt.plot(x_values, y_values, color='red')

plt.title('LDA Separating Line')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

