import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import seaborn as sns
colors = sns.color_palette("colorblind")



# Open the file containing the copied text output from the automated_test.ipynb script
with open('best_shape.txt', 'r') as file:
    file_content = file.read()

# Using regex we extract the grids size, block size, number of z layers, iterations and runtime
# same as in best_optimization.py
pattern = (
    r"stencil2d-([a-zA-Z0-9\-]+)\.x\+orig|"
    r"(\[\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*[\d\.E\+\-]+\s*\])|"
    r"--size_i\s+(\d+)\s+--size_j\s+(\d+)\s+--"
)

matches = re.findall(pattern, file_content)

results = []

# Process matches with special handling for 'kblocking' as it does not contain block size information
for match in matches:
    for group in match:
        if group:
            results.append(group)
            if 'kblocking' in group:
                # Default hit for kblocking
                results.append('8')
                results.append('8')



# 
reference_time = 9.092780 # Runtime of kblocking with a 1024x1024, num_iter=256, nz=16 


versions = ['ijblocking-math'] # load in only the optimal version
results_dict = {key: None for key in versions}

def string_to_array(conv_string):
    # made to convert '[1,2,3]' to np.array([1,2,3])
    conv_string = conv_string.strip('[]')
    elements = conv_string.split(',')

    return  np.array([float(element) for element in elements])

for i in range(0,4,4):
    results_dict.update({results[i]: [np.concatenate((np.array((int(results[i+1]),int(results[i+2]))),string_to_array(results[i+3])))]})

for i in range(4,324,4):
    results_dict[results[i]].append(np.concatenate((np.array((int(results[i+1]),int(results[i+2]))),string_to_array(results[i+3]))))


for key in results_dict.keys():
    results_dict[key] = np.array(results_dict[key])

# set results to the numpy array of the optimal version
results = results_dict['ijblocking-math']

# make plots for comparing size and shape of the block-size
plt.scatter(results[:,0]*results[:,1],results[:,-1]/reference_time, c = np.log(results[:,0]/results[:,1]),cmap='coolwarm')
    
plt.xlabel('Size of ij block')
plt.ylabel('Normalised runtime')
plt.loglog()
plt.colorbar(label =r'$\log\left(\frac{side_i}{side_j}\right)$')
plt.savefig('Best_shape_size.pdf',bbox_inches = 'tight')
plt.close('all')




k=0
fig, ax = plt.subplots()
for i in [16,32,64,128,256]:
        mask1 = np.where(results[:,0]*results[:,1] == i**2)
        mask2 = np.intersect1d(np.where(results[:,0] == i), np.where(results[:,1] == i))
        if i == 64:
                print(results[mask1,0][0],results[mask1,1][0],results[mask1,0][0]/results[mask1,1][0])
        ax.plot(results[mask1,0][0]/results[mask1,1][0],(results[mask1,-1][0]/results[mask2,-1][0]),'.-', color = colors[k], label = f'{i}x{i}')
        k +=1
        
ax.legend()
ax.set_xlabel('size_i / size_j')
ax.set_ylabel('Normalised runtime')
ax.loglog()
xticks = np.concatenate((np.array([(1/2**x) for x in range(9)]), np.array([(2**x) for x in range(1, 9)])))
print((np.array([(1/2**x) for x in range(9)])))
xtick_labels = [1,0.5,0.25,0.125,0.063,0.31,0.016,0.008,0.004, 2, 4, 8, 16, 32, 64, 128, 256]
ax.set_xticks(xticks)

ax.set_xticklabels(xtick_labels)

plt.xticks(rotation=45)
plt.savefig('Best_shape.pdf',bbox_inches = 'tight')
