import numpy as np
import matplotlib.pyplot as plt


p = 0
def prob(relevant, irrelevant, select):
    p = 1 - relevant/(relevant+irrelevant)
    p_none = np.power(p, select)
    at_least_one = 1 - p_none
    return at_least_one


data = np.zeros((10,10))
for i in range(1, 11):
    for j in range(1, 11):
        select = int(np.floor(np.sqrt(j*10)))
        data[-1+i,-1+j] = prob(i,j*10,select)


fig, ax = plt.subplots()
plt.gray()
cs = ax.imshow(data, extent=[10,100,10,1])
ax.set_aspect(10)
plt.xlabel('Irrelevant Features')
plt.ylabel('Relevant Features')
plt.title('Probability to select at least one relevant feature')
fig.colorbar(cs)