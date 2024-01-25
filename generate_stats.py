# Single Author info:
# mshaikh2 Mushtaq Ahmed Shaikh

#This is a script file which does following things
#Combine the files of same message size and same scenrio for different ranks
#For ex scenario 2 has rank0 and rank 1 for node 0; Their contents will be concatenated into a single file 
#Generate stats file which is a list of list of dicts. 
#There are 7 lists for each message size and each list within contains a list of 5 dictionaries.
#The keys of the dict are mean q1 q3 min max

import os
import re
import sys
import numpy as np
from scipy import stats

def combine_and_delete():
    #function to combine files with above logic and place it in the directory
    directory = "csv_plot_directory/"
    files = os.listdir(directory)

    # Group files based on common attributes
    file_groups = {}
    for file_name in files:
        match = re.match(r'output_(inter|intra)_(\d+)_size_(\d+)_rank_(\d+)\.csv', file_name)
        if match:
            file_type, message_size, cores, rank = match.groups()
            key = f'output_{file_type}_{message_size}_size_{cores}.csv'
            if key not in file_groups:
                file_groups[key] = {'files': [], 'delete': []}
            file_groups[key]['files'].append(file_name)
            file_groups[key]['delete'].append(file_name)

    # Combine and delete files
    for key, group in file_groups.items():
        combined_values = []
        for file_name in group['files']:
            with open(os.path.join(directory, file_name), 'r') as file:
                values = [float(line.strip()) for line in file]
                combined_values.extend(values)

        # Write combined values to a new file
        output_file = os.path.join(directory, key)
        with open(output_file, 'w') as file:
            for value in combined_values:
                file.write(f'{value}\n')

        # Delete old files
        for file_name in group['delete']:
            os.remove(os.path.join(directory, file_name))
        #print(f'Files combined and deleted for: {key}')

def generate_stats():
    #function to generate stats file by parsing each file contents. Ultimately there will be 35 files in the directory which will be parsed to generate stats file
    # Directory containing the CSV files
    csv_directory = "csv_plot_directory"

    # List of message sizes
    message_sizes = [32768, 65536, 131072, 262144, 524288, 1048576, 2097152]

    # Open stats.txt file for writing
    with open("stats.txt", 'w') as stats_file:
        # Redirect stdout to the stats.txt file
        sys.stdout = stats_file
        final_list = []
        # Iterate over each message size
        for size in message_sizes:
            # Filter files for the current message size
            files_for_size = [file for file in os.listdir(csv_directory) if f"_{size}_" in file]

            # Sort files based on their names
            sorted_files = sorted(files_for_size)

            # Create a list to store the values from each file
            all_values = []
            file_names = []  # To store original file names

            # Iterate through each file and collect values
            for file in sorted_files:
                with open(os.path.join(csv_directory, file), 'r') as f:
                    
                    values = np.loadtxt(f, delimiter=',', skiprows=1)
                    all_values.append(values)
                    file_names.append(file)  # Store original file name

            # store quartiles and box plot statistics in the dict
            #print(f"Message Size: {size} Bytes")
            result_list = []
            for file_name, values in zip(file_names, all_values):
                # Calculate quartiles using np.percentile
                min_val = np.min(values)
                q1 = np.percentile(values, 25)
                median = np.median(values)
                q3 = np.percentile(values, 75)
                max_val = np.max(values)
                mean = np.mean(values)

                result_dict= {
                "med": mean,
                "q1": q1,
                "q3": q3,
                "whislo": min_val,
                "whishi": max_val
                }
                result_list.append(result_dict)

                #print(f"File {file_name} - Quartiles: [min {min_val}, q1 {q1}, median {median}, q3 {q3}, max {max_val}]")

            #print(result_list)
            final_list.append(result_list)
            
            # Box plot statistics (use the first file's values for describe)
            #box_stats = np.percentile(all_values[0], [0, 25, 50, 75, 100])
            #print(f"Box Plot Statistics - Mean: {np.mean(all_values[0])}, Variance: {np.var(all_values[0])}, Skewness: {stats.skew(all_values[0])}")
            #print(f"Box Plot Quartiles: [min {box_stats[0]}, q1 {box_stats[1]}, median {box_stats[2]}, q3 {box_stats[3]}, max {box_stats[4]}]")

            #print("\n" + "="*30 + "\n")

        # Reset stdout back to the original stdout
        print(final_list)
        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    combine_and_delete()
    generate_stats()

