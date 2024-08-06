import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the uploaded Excel file
file_path = 'kmeans_5d_s_2.csv'
data = pd.read_csv(file_path)

# Define a function to extract velocity from the 'Clusters' dictionary column
def extract_velocity(cluster_data):
    clusters = eval(cluster_data)  # Convert string representation of dict to actual dict
    if 0 in clusters:
        return clusters[0]['velocity']
    elif 1 in clusters:
        return clusters[1]['velocity']
    elif 2 in clusters:
        return clusters[2]['velocity']
    return None

# Extract velocities and identify cluster labels
velocities = data['Clusters'].apply(extract_velocity)
cluster_labels = data['Clusters'].apply(lambda x: eval(x).keys()).apply(lambda keys: list(keys)[0])

# Filter data by cluster label
data_0 = data[cluster_labels == 0]
data_1 = data[cluster_labels == 1]
data_2 = data[cluster_labels == 2]

# Extract velocities for each cluster
velocities_0 = velocities[cluster_labels == 0]
velocities_1 = velocities[cluster_labels == 1]
velocities_2 = velocities[cluster_labels == 2]

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(data_0['Instance'], velocities_0, c='blue', label='Cluster_0', alpha=0.6)
plt.scatter(data_1['Instance'], velocities_1, c='green', label='Cluster_1', alpha=0.6)
plt.scatter(data_2['Instance'], velocities_2, c='red', label='Cluster_2', alpha=0.6)

# Plotting the threshold line at velocity = 1
plt.axhline(y=1.5, color='red', linestyle='-', linewidth=0.5)
plt.text(x=1800, y=1.51, s='Threshold line', color='red', fontsize=10)

# Labeling the axes
plt.xlabel('Instance ID')
plt.ylabel('Velocity')

# Adding legend
plt.legend()

# Display the plot
plt.show()
