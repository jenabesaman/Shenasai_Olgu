import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# Assuming you have a CSV file "pima-indians-diabetes.csv"
df = pd.read_csv('pima-indians-diabetes.csv')

# حذف سطرهایی که مقدار ستون سوم ,ششم آنها برابر با 0 است
df = df[df.iloc[:, 2] != 0]
df = df[df.iloc[:, 5] != 0]
# print(df)

# The columns in your CSV file would be:
df.columns = [
    'Number of times pregnant','Plasma glucose concentration a 2 hours in an oral glucose tolerance test',
    'Diastolic blood pressure (mm Hg)','Triceps skinfold thickness (mm)','2-Hour serum insulin (mu U/ml)',
    'Body mass index (weight in kg/(height in m)^2)','Diabetes pedigree function','Age (years)','Class variable (0 or 1)'
]
features = ['Diastolic blood pressure (mm Hg)', 'Body mass index (weight in kg/(height in m)^2)']
target = 'Class variable (0 or 1)'

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# for i, feature in enumerate(features):
#     # Calculate ROC curve and ROC area for each class
#     fpr, tpr, _ = roc_curve(df[target], df[feature])
#     roc_auc = auc(fpr, tpr)
#     axs[i].plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#     axs[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     axs[i].set_xlim([0.0, 1.0])
#     axs[i].set_ylim([0.0, 1.05])
#     axs[i].set_xlabel('False Positive Rate')
#     axs[i].set_ylabel('True Positive Rate')
#     axs[i].set_title('ROC for ' + feature)
#     axs[i].legend(loc="lower right")
# plt.tight_layout()
# plt.show()


# #ب)

# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)
#
# # Apply LDA
# lda = LinearDiscriminantAnalysis()
# lda.fit(X_train, y_train)
# X_train_lda = lda.transform(X_train)
# X_test_lda = lda.transform(X_test)
#
# # Plot the scatter plot
# plt.scatter(X_train_lda[:, 0], X_train_lda[:, 0], c=y_train, cmap='rainbow')
# plt.xlabel('LD1')
# plt.ylabel('LD2')
# plt.title('LDA Scatter Plot')
# plt.show()




# حل قسمت ب روش دوم
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)

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
plt.scatter(lda_result, np.zeros_like(lda_result), c=target, cmap='viridis', marker='o', edgecolor='k')
plt.title('Data Projection on the Linear Discriminant')
plt.xlabel('Linear Discriminant Axis')
plt.yticks([])
plt.show()

threshold = np.mean(lda_result)
print(f'Threshold: {threshold}')