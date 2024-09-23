# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:56:45 2024

@author: angel
"""

# In[import packages]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In[Import Data files]:
#C:\Users\angel\Documents\Studium\MastersDegree\Semester2\High Performance Computing for Weather and Climate\
df = pd.read_csv("domain_size_IsendIrecv_and_overlapping.csv", sep = ';')
df = round(df, 3)
# In[Plotting all in one]
df.iloc[:,1:].plot(alpha=1, cmap='nipy_spectral')#, 'turbo' cmap = 'PiYG') #'gnuplot')
plt.grid(True)
plt.title('Computation Time for Different Domain Sizes')
plt.xlabel('Number of Ranks')
plt.ylabel('Computation Time [s]')
plt.yscale('log')
plt.legend(loc = 1, ncols = 4, fontsize=5, title = 'Domain Size', title_fontsize=6)
plt.xticks(np.arange(0, 25, step=5), labels=np.arange(1,26, step=5))

#plt.savefig('Ranks_vs_DomainSize_IsendIrecv_and_overlapping_logScale.png', dpi = 300, transparent = True)

# In[Plotting differences]
df1 = pd.read_csv("domain_size_IsendIrecv.csv", sep = ';')
df2 = pd.read_csv("domain_size_overlapping.csv", sep = ';')
col1 = df1.iloc[:,0]
diff = df1.iloc[:,1:]-df2.iloc[:,1:]
diff = round(diff, 3)
#df2 = df2.iloc[:,1:]
df_diff = pd.concat([col1, diff], axis=1)
df_diff_3 = df_diff.iloc[:,:-2]

df_diff.iloc[:,1:].plot(alpha=1, cmap='nipy_spectral')#, copper   'nipy_spectral'  'turbo' cmap = 'PiYG') #'gnuplot')
plt.grid(True)
plt.title('Difference of Computation Times (IsendIrecv - Overlapping)')
plt.xlabel('Number of Ranks')
plt.ylabel('Difference [s]')
plt.legend(ncols = 1, title = 'Domain Size')#, fontsize=5.5)
plt.xticks(np.arange(0, 25, step=5), labels=np.arange(1,26, step=5))
#plt.savefig('Ranks_vs_DomainSize_Difference.png', dpi = 300, transparent = True)

