# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:25:20 2024

@author: angel

To plot the outputs from the communication strategies, actualize the csv file
To get the data: run the for loop on the CSCS jupyter hub
"""

# In[import packages]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In[Import Data files]:
#C:\Users\angel\Documents\Studium\MastersDegree\Semester2\High Performance Computing for Weather and Climate\
df = pd.read_csv("rank_vs_computationTime_1node.csv", sep = ';')
df = df.loc[1:,:]
df = round(df, 3)

# In[Plotting]
df.plot(x='rank', y=df.columns[1:], alpha = 0.8, cmap = 'PiYG') #'gnuplot')
plt.grid(True)
plt.title('Computation Time of Communication Strategies on 1 Node')
plt.xlabel('Number of Ranks')
plt.ylabel('Computation Time [s]')
plt.legend(loc = 1, ncols = 2, title = 'Communication Strategy')
#plt.savefig('Ranks_vs_ComputationTime.png', dpi = 300, transparent = True)
