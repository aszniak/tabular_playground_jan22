import numpy as np
import matplotlib.pyplot as plt

days = np.linspace(0, 365, 365)
plt.plot(np.sin(2 * np.pi * days/365))
plt.show()


