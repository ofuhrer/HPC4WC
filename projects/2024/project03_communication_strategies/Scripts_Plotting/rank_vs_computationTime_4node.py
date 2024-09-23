# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 07:59:51 2024

@author: angel
"""
# In[import packages]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In[Import Data files]:
#C:\Users\angel\Documents\Studium\MastersDegree\Semester2\High Performance Computing for Weather and Climate\
df = pd.read_csv("rank_vs_computationTime_4nodes.csv", sep = ';')
df = df.drop('send_Irecv_2hp', axis=1)
df = round(df, 3)

# In[Plotting]
df.plot(x='rank', y=df.columns[1:], alpha = 0.7)#, cmap = 'blues') #'gnuplot')
plt.grid(True)
plt.title('Computation Time of Communication Strategies on 4 Nodes')
plt.xlabel('Number of Ranks')
plt.ylabel('Computation Time [s]')
plt.legend(loc = 1, ncols = 2, title = 'Communication Strategy')
#plt.savefig('Ranks_vs_ComputationTime_4nodes.png', dpi = 300, transparent = True)
