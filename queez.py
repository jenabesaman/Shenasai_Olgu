import numpy as np

# داده ها
w1 = np.array([(0,0), (0,1), (2,2), (3,1), (3,2), (3,3)])
w2 = np.array([(6,9), (8,9), (9,8), (9,9), (9,10), (8,11)])

# محاسبه میانگین هر دسته
m1 = np.mean(w1, axis=0)
m2 = np.mean(w2, axis=0)

# محاسبه S1 و S2
S1 = np.dot((w1 - m1).T, (w1 - m1))
S2 = np.dot((w2 - m2).T, (w2 - m2))

# محاسبه Sw
Sw = S1 + S2

# محاسبه w
w = np.dot(np.linalg.inv(Sw), (m2 - m1))

print("w:", w)