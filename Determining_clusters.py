import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("ds1_balanced.csv")

# Fix column names (strip spaces and lowercase for consistency)
df.columns = df.columns.str.strip()
print("Column names in dataset:", df.columns.tolist())  # Print actual column names

# Define expected attributes
expected_attributes = ['soil_moisture', 'temperature', 'humidity', 'ph', 'rainfall']

# Map actual columns (case-insensitive handling)
actual_columns = {col.lower(): col for col in df.columns}  
attributes = [actual_columns[attr.lower()] if attr.lower() in actual_columns else None for attr in expected_attributes]

# Remove missing attributes
attributes = [attr for attr in attributes if attr is not None]
print("Processing attributes:", attributes)  # Ensure all attributes are included

if not attributes:
    raise ValueError("None of the specified attributes were found in the dataset!")

optimal_clusters_dict = {}  # Store results

# Loop through attributes
for attribute in attributes:
    print(f"\nProcessing: {attribute}")

    data = df[[attribute]].values  
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    max_clusters = 10
    fpc_values, pe_values = [], []
    cluster_range = range(2, max_clusters + 1)

    for n_clusters in cluster_range:
        # Apply Gaussian Mixture Model (GMM)
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(data_scaled)
        
        # Get the fuzzy membership (responsibilities)
        responsibilities = gmm.predict_proba(data_scaled)
        
        # Calculate the FPC (Fuzzy Partition Coefficient)
        fpc = np.sum(responsibilities**2) / len(data_scaled)
        fpc_values.append(fpc)  # Store FPC value
        
        # Calculate the Partition Entropy (PE)
        pe = -np.sum(responsibilities * np.log2(responsibilities + 1e-10)) / len(data_scaled)
        pe_values.append(pe)  # Store PE value

    # Normalize FPC & PE
    fpc_norm = (fpc_values - np.min(fpc_values)) / (np.max(fpc_values) - np.min(fpc_values))
    pe_norm = (pe_values - np.min(pe_values)) / (np.max(pe_values) - np.min(pe_values))

    # Find intersection
    intersection_point = None
    for i in range(len(cluster_range) - 1):
        if (fpc_norm[i] - pe_norm[i]) * (fpc_norm[i + 1] - pe_norm[i + 1]) < 0:
            x1, x2 = cluster_range[i], cluster_range[i + 1]
            y1_fpc, y2_fpc = fpc_norm[i], fpc_norm[i + 1]
            y1_pe, y2_pe = pe_norm[i], pe_norm[i + 1]

            slope_fpc = (y2_fpc - y1_fpc) / (x2 - x1)
            slope_pe = (y2_pe - y1_pe) / (x2 - x1)
            intersection_x = (y1_pe - y1_fpc + slope_fpc * x1 - slope_pe * x1) / (slope_fpc - slope_pe)
            intersection_y = y1_fpc + slope_fpc * (intersection_x - x1)
            intersection_point = (intersection_x, intersection_y)
            break

    # Determine optimal clusters
    optimal_clusters = int(round(intersection_point[0])) if intersection_point else None
    optimal_clusters_dict[attribute] = optimal_clusters
    print(f"Optimal clusters for {attribute}: {optimal_clusters}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(cluster_range, fpc_norm, marker='o', linestyle='-', color='b', label='FPC (Normalized)')
    plt.plot(cluster_range, pe_norm, marker='o', linestyle='-', color='g', label='PE (Normalized)')
    plt.title(f'FPC & PE for {attribute}')
    plt.xlabel('Clusters')
    plt.ylabel('Normalized Value')
    plt.xticks(cluster_range)
    plt.grid()
    
    if intersection_point:
        plt.scatter(intersection_point[0], intersection_point[1], color='r', label=f'Intersection (k={intersection_point[0]:.2f})')
        plt.axvline(x=optimal_clusters, color='m', linestyle=':', label=f'Optimal k={optimal_clusters}')

    plt.legend()
    plt.show()

    # Run Gaussian Mixture Model (GMM) for optimal clusters
    if optimal_clusters:
        gmm = GaussianMixture(n_components=optimal_clusters, random_state=42)
        gmm.fit(data_scaled)
        cluster_centers = scaler.inverse_transform(gmm.means_)
        print(f"Cluster Centers for {attribute}:\n", cluster_centers)

        cluster_membership = gmm.predict(data_scaled)
        df[f'{attribute}_cluster'] = cluster_membership

        # Print cluster analysis
        print(f"\nCluster Analysis for {attribute}:")
        print(df.groupby(f'{attribute}_cluster')[attribute].describe())

        # Visualize clusters
        plt.figure(figsize=(8, 6))
        plt.scatter(range(len(data_scaled)), data_scaled, c=cluster_membership, cmap='viridis', marker='o')
        plt.title(f'Clustering ({attribute}, k={optimal_clusters})')
        plt.xlabel('Data Index')
        plt.ylabel(f'{attribute} (Scaled)')
        plt.colorbar(label='Cluster')
        plt.show()

# Print summary
print("\nFinal Optimal Clusters:")
for attr, clusters in optimal_clusters_dict.items():
    print(f"{attr}: {clusters}")