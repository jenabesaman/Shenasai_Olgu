import pandas as pd

# Assuming you have a CSV file "data.csv"
df = pd.read_csv('pima-indians-diabetes.csv')

# The columns in your CSV file would be:
df.columns = [
    'Number of times pregnant',
    'Plasma glucose concentration a 2 hours in an oral glucose tolerance test',
    'Diastolic blood pressure (mm Hg)',
    'Triceps skinfold thickness (mm)',
    '2-Hour serum insulin (mu U/ml)',
    'Body mass index (weight in kg/(height in m)^2)',
    'Diabetes pedigree function',
    'Age (years)',
    'Class variable (0 or 1)'
]

print(df)


#الف)
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# فرض می‌کنیم 'df' DataFrame شما و 'y' متغیر هدف شما است
features = ['Diastolic blood pressure (mm Hg)', 'Body mass index (weight in kg/(height in m)^2)']
target = 'Class variable (0 or 1)'

for feature in features:
    # محاسبه نمودار ROC و مساحت ROC برای هر کلاس
    fpr, tpr, _ = roc_curve(df[target], df[feature])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for' + feature)
    plt.legend(loc="lower right")
    plt.show()

#ب)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import numpy as np

# فرض می‌کنیم 'df' DataFrame شما و 'y' متغیر هدف شما است
target = 'Class variable (0 or 1)'

# حذف نمونه‌هایی که دارای مقدار صفر هستند
df = df.replace(0, np.nan)
df = df.dropna()

# تعیین ویژگی‌ها و هدف
X = df.drop(target, axis=1)
y = df[target]

# تعداد کلاس‌ها و ویژگی‌ها را محاسبه می‌کنیم
n_classes = len(np.unique(y))
n_features = X.shape[1]

# اعمال LDA
n_components = min(n_features, n_classes - 1)
lda = LDA(n_components=n_components)
X_lda = lda.fit_transform(X, y)

# رسم نمودار
plt.scatter(X_lda, y)
plt.xlabel('LDA1')
plt.ylabel(target)
plt.show()
