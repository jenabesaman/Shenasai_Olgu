import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# Assuming you have a CSV file "pima-indians-diabetes.csv"
df = pd.read_csv('pima-indians-diabetes.csv')
# The columns in your CSV file would be:
df.columns = [
    'Number of times pregnant','Plasma glucose concentration a 2 hours in an oral glucose tolerance test',
    'Diastolic blood pressure (mm Hg)','Triceps skinfold thickness (mm)','2-Hour serum insulin (mu U/ml)',
    'Body mass index (weight in kg/(height in m)^2)','Diabetes pedigree function','Age (years)','Class variable (0 or 1)'
]
features = ['Diastolic blood pressure (mm Hg)', 'Body mass index (weight in kg/(height in m)^2)']
target = 'Class variable (0 or 1)'

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

for i, feature in enumerate(features):
    # Calculate ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(df[target], df[feature])
    roc_auc = auc(fpr, tpr)
    axs[i].plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    axs[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[i].set_xlim([0.0, 1.0])
    axs[i].set_ylim([0.0, 1.05])
    axs[i].set_xlabel('False Positive Rate')
    axs[i].set_ylabel('True Positive Rate')
    axs[i].set_title('ROC for ' + feature)
    axs[i].legend(loc="lower right")
plt.tight_layout()
plt.show()


#ب)

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('pima-indians-diabetes.csv')

# Drop columns with zero values
df = df.loc[:, (df != 0).any(axis=0)]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1],
                                                    test_size=0.2, random_state=42)

# Apply LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
X_train_lda = lda.transform(X_train)
X_test_lda = lda.transform(X_test)

# Plot the scatter plot
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 0], c=y_train, cmap='rainbow')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('LDA Scatter Plot')
plt.show()