import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

colors = sns.color_palette("colorblind")


# Open the file with the copied output from automated tests
with open('best_optimization.txt', 'r') as file:
    file_content = file.read()
    
# Using regex we extract the grids size, block size, number of z layers, iterations and runtime
# same as in best_shape.py
pattern = (
    r"stencil2d-([a-zA-Z0-9\-]+)\.x\+orig|"
    r"(\[\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*[\d\.E\+\-]+\s*\])|"
    r"--size_i\s+(\d+)\s+--size_j\s+(\d+)\s+--"
)

matches = re.findall(pattern, file_content)

results = []

# Process matches and add special handling for 'kblocking'
for match in matches:
    for group in match:
        if group:
            results.append(group)
            if 'kblocking' in group:
                # Add a default hit for kblocking
                results.append('8')
                results.append('8')



# load in all versions to compare.
versions = ['kblocking', 'ijblocking-small', 'ijblocking-inline', 'ijblocking-math', 'ijblocking2']
results_dict = {key: None for key in versions}

def string_to_array(conv_string):
    conv_string = conv_string.strip('[]')
    elements = conv_string.split(',')

    return  np.array([float(element) for element in elements])

for i in range(0,20,4):
    results_dict.update({results[i]: [np.concatenate((np.array((int(results[i+1]),int(results[i+2]))),string_to_array(results[i+3])))]})

for i in range(20,120,4):
    results_dict[results[i]].append(np.concatenate((np.array((int(results[i+1]),int(results[i+2]))),string_to_array(results[i+3]))))


for key in results_dict.keys():
    results_dict[key] = np.array(results_dict[key])


# Compare the different versions
for i, ver in enumerate(versions):
    plt.plot(results_dict[versions[i]][:,3],results_dict[versions[i]][:,-1]/results_dict[versions[0]][:,-1], '.-',label = ver,color = colors[i])
    
plt.legend()
plt.yscale('log')
plt.xticks([2**x for x in range(6,12)],rotation = 75)
plt.xlabel('Side of the square')
plt.ylabel('Normalised runtime')
plt.savefig('Math_is_best.pdf', bbox_inches='tight')
