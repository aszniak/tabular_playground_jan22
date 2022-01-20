import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# gdp_df = pd.read_csv('gdp.csv')
#
# gdp_df = gdp_df[gdp_df.year == 2015]
# print(gdp_df['Finland'].item())



arr = np.load('train_data_array.npy')
for col in range(len(arr[0])):
    if np.nanmax(arr[:, col]) > 1:
        print(col)

