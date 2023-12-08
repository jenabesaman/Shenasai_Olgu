import numpy as np
import matplotlib.pyplot as plt

# تعریف کرنل مثلثی
def triangular_kernel(u):
    return np.where(np.abs(u) <= 1, 1 - np.abs(u), 0)

# تعریف تابع چگالی احتمال با استفاده از روش پنجره پارزن و کرنل مثلثی
def parzen_window(x, data, h):
    n = len(data)
    return (1/n) * np.sum(triangular_kernel((x - data) / h))

# داده‌ها
data = np.array([1, 4, 3, 4, 0, 2, 4, 2])
data.sort()

# تعیین عرض پنجره
h = 1.0  # این مقدار را می‌توانید بر اساس نیاز خود تنظیم کنید

# محاسبه تابع چگالی احتمال برای یک بازه از مقادیر x
x_values = np.linspace(np.min(data), np.max(data), 1000)
y_values = [parzen_window(x, data, h) for x in x_values]

# رسم تابع چگالی احتمال
plt.plot(x_values, y_values)
#تابع چگالی احتمال با استفاده از روش پنجره پارزن و کرنل مثلثی
plt.title('Probability density function using Parzen window method and triangular kernel')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()