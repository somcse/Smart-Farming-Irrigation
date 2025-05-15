import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# Load dataset
df = pd.read_csv("ds1_balanced.csv")

# Clean column names
df.columns = df.columns.str.strip()
print("Columns:", df.columns.tolist())

# Define expected attributes
expected_attributes = ['soil_moisture', 'temperature', 'humidity', 'ph', 'rainfall']

# Map actual columns
actual_columns = {col.lower(): col for col in df.columns}
attributes = [actual_columns[attr.lower()] for attr in expected_attributes if attr.lower() in actual_columns]
if not attributes:
    raise ValueError("None of the expected attributes found.")

# Detect pump column
pump_col = None
for col in df.columns:
    if 'pump' in col.lower() and df[col].nunique() <= 2:
        pump_col = col
        break
if pump_col is None:
    raise ValueError("Pump status column not found!")

print(f"Pump status column detected: {pump_col}")

# Cluster labels
label_mappings = {
    2: ['low', 'high'],
    3: ['low', 'mid', 'high'],
    4: ['low', 'medium', 'high', 'very high'],
    6: ['very low', 'low', 'mid', 'high', 'very high', 'extremely high'],
    7: ['extremely low', 'very low', 'low', 'mid', 'high', 'very high', 'extremely high']
}

# Store fuzzy data
fuzzy_data = pd.DataFrame()

for attribute in attributes:
    print(f"\nProcessing: {attribute}")
    
    data = df[[attribute]].values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    max_clusters = 10
    fpc_values, pe_values = [], []
    cluster_range = range(2, max_clusters + 1)

    for n_clusters in cluster_range:
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(data_scaled)
        responsibilities = gmm.predict_proba(data_scaled)
        fpc = np.sum(responsibilities**2) / len(data_scaled)
        pe = -np.sum(responsibilities * np.log2(responsibilities + 1e-10)) / len(data_scaled)
        fpc_values.append(fpc)
        pe_values.append(pe)

    fpc_norm = (fpc_values - np.min(fpc_values)) / (np.max(fpc_values) - np.min(fpc_values))
    pe_norm = (pe_values - np.min(pe_values)) / (np.max(pe_values) - np.min(pe_values))

    intersection_point = None
    for i in range(len(cluster_range) - 1):
        if (fpc_norm[i] - pe_norm[i]) * (fpc_norm[i + 1] - pe_norm[i + 1]) < 0:
            x1, x2 = cluster_range[i], cluster_range[i + 1]
            y1_fpc, y2_fpc = fpc_norm[i], fpc_norm[i + 1]
            y1_pe, y2_pe = pe_norm[i], pe_norm[i + 1]

            slope_fpc = (y2_fpc - y1_fpc) / (x2 - x1)
            slope_pe = (y2_pe - y1_pe) / (x2 - x1)

            intersection_x = (y1_pe - y1_fpc + slope_fpc * x1 - slope_pe * x1) / (slope_fpc - slope_pe)
            intersection_point = (intersection_x, y1_fpc + slope_fpc * (intersection_x - x1))
            break

    optimal_clusters = int(round(intersection_point[0])) if intersection_point else 3
    print(f"Optimal clusters for {attribute}: {optimal_clusters}")

    # Fit GMM
    gmm = GaussianMixture(n_components=optimal_clusters, random_state=42)
    gmm.fit(data_scaled)
    cluster_centers = scaler.inverse_transform(gmm.means_).flatten()

    cluster_labels = gmm.predict(data_scaled)
    sorted_indices = np.argsort(cluster_centers)
    label_map = {original: i for i, original in enumerate(sorted_indices)}

    if optimal_clusters in label_mappings:
        category_labels = label_mappings[optimal_clusters]
    else:
        category_labels = [f"cluster_{i}" for i in range(optimal_clusters)]

    categorical_labels = [category_labels[label_map[cl]] for cl in cluster_labels]
    fuzzy_data[f'{attribute}_fuzzy'] = categorical_labels

    # Print cluster info
    print(f"Cluster centers for {attribute}:")
    for i in sorted_indices:
        label_idx = sorted_indices.tolist().index(i)
        label = category_labels[label_idx]
        print(f"  {label}: {cluster_centers[i]:.2f}")

# Add pump status fuzzified
fuzzy_data['Pump Data'] = df[pump_col].apply(lambda x: 'OFF' if x == 0 else 'ON')

# Save final fuzzy dataset
fuzzy_data.to_csv("balanced_fuzzified_dataset.csv", index=False)
print("\nFuzzified dataset saved as 'balanced_fuzzified_dataset.csv'")
print(fuzzy_data.head())