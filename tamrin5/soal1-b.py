import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
# Load the dataset
df = pd.read_csv('pima-indians-diabetes.csv', header=None, names=['Number of times pregnant', 'Plasma glucose concentration a 2 hours in an oral glucose tolerance test', 'Diastolic blood pressure (mm Hg)', 'Triceps skinfold thickness (mm)', '2-Hour serum insulin (mu U/ml)', 'Body mass index (weight in kg/(height in m)^2)', 'Diabetes pedigree function', 'Age (years)', 'Class variable (0 or 1)'])
# Replace zeros with NaNs in specific columns as they are considered missing values
for column in ['Plasma glucose concentration a 2 hours in an oral glucose tolerance test', 'Diastolic blood pressure (mm Hg)', 'Triceps skinfold thickness (mm)', '2-Hour serum insulin (mu U/ml)', 'Body mass index (weight in kg/(height in m)^2)']:
    df[column].replace(0, np.nan, inplace=True)
# Drop rows with NaNs
df.dropna(inplace=True)
# انتخاب ویژگی‌ها و هدف
features = df.drop('Class variable (0 or 1)', axis=1)
target = df['Class variable (0 or 1)']
# استانداردسازی ویژگی‌ها
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
# اعمال LDA
lda = LinearDiscriminantAnalysis(n_components=1)
lda_result = lda.fit_transform(features_scaled, target)
# تصویر نمودن داده‌ها بر روی یک خط با بیشترین جداپذیری
plt.figure(figsize=(8, 6))
plt.scatter(lda_result[:, 0], np.zeros_like(lda_result), c=target, cmap='viridis', marker='o', edgecolor='k')
plt.title('Data Projection on the Linear Discriminant')
plt.xlabel('Linear Discriminant Axis')
plt.yticks([])
plt.show()
threshold = np.mean(lda_result)
print(f'Threshold: {threshold}')
