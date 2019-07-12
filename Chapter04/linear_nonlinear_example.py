import matplotlib.pyplot as plt
import numpy as np

x = [i for i in range(100)]
y = [5 for i in range(100)]

for i in range(30, 60):
    y[i] = 4+((i-45)**2)/230

for i in range(100):
    y[i] = y[i] + np.random.uniform(-0.03, 0.03)

plt.scatter(x, y, label='Data')

y = [5 for i in range(100)]

for i in range(20, 70):
    y[i] = 4+((i-45)**2)/230

plt.plot([5 for i in range(100)], label='Linear $y=5$', color='C1')
plt.plot(x[20:70], y[20:70], label='Non-Linear $y=x^2$', color='C2')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear and Non-Linear Relationships')
plt.legend()

