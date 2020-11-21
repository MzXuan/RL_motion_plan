import numpy as np
from matplotlib import pyplot as plt

x = 0.1*np.arange(-10,10)
y = np.arctan(2*(x-(-0.1)))
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x,y)
plt.show()