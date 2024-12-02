import os
import yaml
import json
import pandas as pd

# Base folder where the subfolders 0-29 are located
base_folder = "/bigwork/nhmlhuer/git/backup/gaussian_splatting_rl/hydra/multirun/2024-10-18/16-23-13"

# List to store the results as dictionaries for each subfolder
results_list = []

# Iterate through subfolders 0-29 (skip .submitit)
for subfolder in range(30):
    subfolder_path = os.path.join(base_folder, str(subfolder))
    
    if ".submitit" in subfolder_path:
        continue
    
    # Search in each numbered folder inside /output/ for the specific files
    output_folder = os.path.join(subfolder_path, 'output')
    if not os.path.exists(output_folder):
        continue

    # Find folders that start with a number
    for folder in os.listdir(output_folder):
        if not folder[0].isdigit():
            continue
        
        numbered_folder = os.path.join(output_folder, folder)
        
        # Dictionary to store extracted data from the current subfolder
        subfolder_results = {
            "subfolder": subfolder,
            "folder": folder
        }

        # Search for the dataset dynamically based on file names
        dataset = None
        
        # Look for any matching dataset
        for file in os.listdir(numbered_folder):
            if "gaussian_num_points_" in file:
                dataset = file.split("gaussian_num_points_")[-1].split(".")[0]
                subfolder_results['dataset'] = dataset  # Add the dataset to results
                break
        
        # Paths to the required files
        if dataset:
            gaussian_file = os.path.join(numbered_folder, f'gaussian_num_points_{dataset}.txt')
            per_view_file = os.path.join(numbered_folder, f'per_view_{dataset}.json')
            results_file = os.path.join(numbered_folder, f'results_{dataset}.json')

            # Read gaussian_num_points_<dataset>.txt
            if os.path.exists(gaussian_file):
                with open(gaussian_file, 'r') as f:
                    num_points = f.readline().strip().split(":")[-1].strip()
                    subfolder_results['gaussian_num_points'] = num_points

            # Read per_view<dataset>.json
            #if os.path.exists(per_view_file):
            #    with open(per_view_file, 'r') as f:
            #        per_view_data = json.load(f)
            #        subfolder_results['per_view'] = per_view_data

            # Read results_<dataset>.json
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                    subfolder_results['results'] = results_data

        # Read the overrides.yaml file in the main subfolder
        overrides_file = os.path.join(subfolder_path, 'overrides.yaml')
        if os.path.exists(overrides_file):
            with open(overrides_file, 'r') as f:
                overrides_data = yaml.safe_load(f)
                # Flatten YAML data and add to subfolder results
                for item in overrides_data:
                    # Split based on the first '=' to avoid issues with '=' in values
                    if isinstance(item, str) and '=' in item:
                        key, value = item.split('=', 1)  # Split only on the first '='
                        subfolder_results[key.strip()] = value.strip()

        # Append the result to the list if we collected any information
        if subfolder_results:
            results_list.append(subfolder_results)

# Convert the results list into a pandas DataFrame
df = pd.DataFrame(results_list)
# Group by 'subfolder' and count the unique 'dataset' values in each subfolder
dataset_counts = df.groupby('subfolder')['dataset'].nunique()

# Filter to get subfolders with exactly 4 datasets
subfolders_with_4_datasets = dataset_counts[dataset_counts == 4].index

# Filter the original DataFrame to include only subfolders with exactly 4 datasets
df_filtered = df[df['subfolder'].isin(subfolders_with_4_datasets)]

# Display the filtered DataFrame
print(df_filtered)
# Save the DataFrame to a CSV or JSON file for future use
#df.to_csv('results_combined.csv', index=False)
#df.to_json('results_combined.json', orient='records', indent=4)

# Display the DataFrame to the user for further analysis
#print("Results combined into a DataFrame and saved to CSV and JSON.")
