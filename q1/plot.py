# Single Author info:
# mshaikh2 Mushtaq Ahmed Shaikh

#This helper script is used to generate boxplot using the stats file generated. 

import matplotlib.pyplot as plt

with open('stats.txt', 'r') as file:
    stats_data = eval(file.read())

# Split the list into seven different lists
num_lists = 7
stats_lists = [lst for lst in stats_data]

# Create variables stats_1, stats_2, ..., stats_7
for i, stats_list in enumerate(stats_lists):
    globals()[f'stats_{i + 1}'] = stats_list
    #print(stats_1)
    #print(globals()[f'stats_{i + 1}'])

fig, ax = plt.subplots(figsize=(20, 10))
positions = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29, 30, 31, 32, 33, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47]
boxplot = ax.bxp(stats_1 + stats_2 + stats_3 + stats_4 + stats_5 + stats_6 + stats_7, showfliers=False, positions=positions, patch_artist=True)

# Change box colors to blue
for box in boxplot['boxes']:
    box.set(color='blue', linewidth=2, facecolor='blue')

plt.xlabel('Message Sizes')
plt.ylabel('Time in Microseconds')
plt.title('Round Trip Time Box Plot')
plt.xticks([3, 10, 17, 24, 31, 38, 45], ["32 KB", "64 KB", "128 KB", "256 KB", "512 KB", "1 MB", "2 MB"])
plt.savefig('p1.png')
plt.show()

