import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("success_rate.csv")
print(data.head())

# x = data['epoch'].values
x = data['train/episode'].values
# mlp_rate = data['epoch'].values
# rnn_rate = data['epoch'].values

# t = np.linspace(0, 1.5*np.pi, 2500)
# # y = np.sin(t**2)+np.random.random(2500)*.6
# # df = pd.DataFrame(y)
plt.plot(data['train/success_rate_1127_mlp'], 'lightblue', linewidth=5)
plt.plot(data['train/success_rate_1127_mlp'].rolling(5).mean(), 'b', linewidth=2)

plt.plot(data['train/success_rate_joint_space'], 'lightcoral', linewidth=5)
plt.plot(data['train/success_rate_joint_space'].rolling(5).mean(), 'r', linewidth=2)
plt.show()