# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 09:53:20 2024

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
df = pd.read_csv("domain_size_IsendIrecv.csv", sep = ';')
#df = df.iloc[:, 1:]
df = round(df, 3)

# In[Plotting]
df.iloc[:,1:].plot(alpha=1, cmap='turbo')#, cmap = 'PiYG') #'gnuplot')
plt.grid(True)
plt.title('Computation Time for Different Domain Sizes')
plt.xlabel('Number of Ranks')
plt.ylabel('Computation Time [s]')
plt.yscale('log')
plt.legend(loc = 1, ncols = 5, fontsize = 7.5, title = 'Domain Size')
#plt.savefig('Ranks_vs_DomainSize_IsendIrecv_logScale.png', dpi = 300, transparent = True)
