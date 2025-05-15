import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "pump dataset.csv"  # Replace with the actual file path
df = pd.read_csv(file_path)

# Define bell-shaped membership function
def bell_mf(x, c, a, b):
    return 1 / (1 + np.abs((x - c) / a) ** (2 * b))

# === 1. Soil Moisture ===
soil_moisture = np.linspace(df["soil_moisture"].min(), df["soil_moisture"].max(), 300)
k_soil = 4
centers_soil = np.linspace(min(soil_moisture), max(soil_moisture), k_soil)
names_soil = ["Low", "Mid", "High", "Very High"]
a_soil = (max(soil_moisture) - min(soil_moisture)) / (2 * k_soil)
b = 2

membership_soil = np.array([bell_mf(soil_moisture, c, a_soil, b) for c in centers_soil])
membership_soil /= np.sum(membership_soil, axis=0)

plt.figure(figsize=(8, 5))
for i, name in enumerate(names_soil):
    plt.plot(soil_moisture, membership_soil[i], label=name)
plt.xlabel("Soil Moisture")
plt.ylabel("Membership Value")
plt.title("Bell-Shaped Fuzzy Partitioning of Soil Moisture")
plt.legend()
plt.show()

# === 2. Air Humidity ===
humidity = np.linspace(df["humidity"].min(), df["humidity"].max(), 300)
k_humidity = 6
centers_humidity = np.linspace(min(humidity), max(humidity), k_humidity)
names_humidity = ["Very Low", "Low", "Mid", "High", "Very High", "Extremely High"]
a_humidity = (max(humidity) - min(humidity)) / (2 * k_humidity)

membership_humidity = np.array([bell_mf(humidity, c, a_humidity, b) for c in centers_humidity])
membership_humidity /= np.sum(membership_humidity, axis=0)

plt.figure(figsize=(8, 5))
for i, name in enumerate(names_humidity):
    plt.plot(humidity, membership_humidity[i], label=name)
plt.xlabel("Air Humidity")
plt.ylabel("Membership Value")
plt.title("Bell-Shaped Fuzzy Partitioning of Air Humidity")
plt.legend()
plt.show()

# === 3. Temperature ===
temperature = np.linspace(df["temperature"].min(), df["temperature"].max(), 300)
k_temp = 4
centers_temp = np.linspace(min(temperature), max(temperature), k_temp)
names_temp = ["Low", "Mid", "High", "Very High"]
a_temp = (max(temperature) - min(temperature)) / (2 * k_temp)

membership_temp = np.array([bell_mf(temperature, c, a_temp, b) for c in centers_temp])
membership_temp /= np.sum(membership_temp, axis=0)

plt.figure(figsize=(8, 5))
for i, name in enumerate(names_temp):
    plt.plot(temperature, membership_temp[i], label=name)
plt.xlabel("Temperature")
plt.ylabel("Membership Value")
plt.title("Bell-Shaped Fuzzy Partitioning of Temperature")
plt.legend()
plt.show()

# === 4. Rainfall ===
rainfall = np.linspace(df["rainfall"].min(), df["rainfall"].max(), 300)
k_rain = 7
centers_rain = np.linspace(min(rainfall), max(rainfall), k_rain)
names_rain = ["Extremely Low", "Very Low", "Low", "Mid", "High", "Very High", "Extremely High"]
a_rain = (max(rainfall) - min(rainfall)) / (2 * k_rain)

membership_rain = np.array([bell_mf(rainfall, c, a_rain, b) for c in centers_rain])
membership_rain /= np.sum(membership_rain, axis=0)

plt.figure(figsize=(8, 5))
for i, name in enumerate(names_rain):
    plt.plot(rainfall, membership_rain[i], label=name)
plt.xlabel("Rainfall")
plt.ylabel("Membership Value")
plt.title("Bell-Shaped Fuzzy Partitioning of Rainfall")
plt.legend()
plt.show()

# === 5. pH ===
ph = np.linspace(df["ph"].min(), df["ph"].max(), 300)
k_ph = 2
centers_ph = np.linspace(min(ph), max(ph), k_ph)
names_ph = ["Low", "High"]
a_ph = (max(ph) - min(ph)) / (2 * k_ph)

membership_ph = np.array([bell_mf(ph, c, a_ph, b) for c in centers_ph])
membership_ph /= np.sum(membership_ph, axis=0)

plt.figure(figsize=(8, 5))
for i, name in enumerate(names_ph):
    plt.plot(ph, membership_ph[i], label=name)
plt.xlabel("pH")
plt.ylabel("Membership Value")
plt.title("Bell-Shaped Fuzzy Partitioning of pH")
plt.legend()
plt.show()
