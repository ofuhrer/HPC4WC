# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 16:57:24 2024

@author: angel
"""
# In[import packages]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In[Import Data files]:
#C:\Users\angel\Documents\Studium\MastersDegree\Semester2\High Performance Computing for Weather and Climate\
df1 = pd.read_csv("rank_vs_computationTime_1-4nodes.csv", sep = ';')
df1 = round(df1, 3)

# In[Plotting]
df1.plot(x='rank', y=df1.columns[1:], alpha = 0.8, cmap = 'rainbow') #'gnuplot')
plt.grid(True)
plt.title('Computation Time of Communication Strategies on 1, 2, and 4 Nodes')
plt.xlabel('Number of Ranks')
plt.ylabel('Computation Time [s]')
plt.legend(loc = 1, ncols = 3, title = 'Communication Strategy', fontsize = 6.5)
#plt.savefig('Ranks_vs_ComputationTime_1-4nodes.png', dpi = 300, transparent = True)
