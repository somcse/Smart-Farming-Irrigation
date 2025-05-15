import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Bell-shaped membership function
def bell_mf(x, a, b, c):
    return 1 / (1 + np.abs((x - c) / a) ** (2 * b))

# Define input variables
soil_moisture = ctrl.Antecedent(np.arange(0, 101, 1), 'soil_moisture')
rainfall = ctrl.Antecedent(np.arange(0, 101, 1), 'rainfall')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
temperature = ctrl.Antecedent(np.arange(0, 51, 1), 'temperature')
ph = ctrl.Antecedent(np.arange(0, 14, 1), 'ph')

# Define output variable
pump = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'pump')

# Define soil moisture membership functions
soil_moisture['very_low'] = bell_mf(soil_moisture.universe, 8, 2, 5)
soil_moisture['low'] = bell_mf(soil_moisture.universe, 10, 2, 20)
soil_moisture['medium'] = bell_mf(soil_moisture.universe, 10, 2, 50)
soil_moisture['high'] = bell_mf(soil_moisture.universe, 10, 2, 80)
soil_moisture['very_high'] = bell_mf(soil_moisture.universe, 8, 2, 95)

# Define rainfall membership functions
rainfall['extremely_low'] = bell_mf(rainfall.universe, 8, 2, 0)
rainfall['very_low'] = bell_mf(rainfall.universe, 10, 2, 15)
rainfall['low'] = bell_mf(rainfall.universe, 10, 2, 30)
rainfall['mid'] = bell_mf(rainfall.universe, 10, 2, 50)
rainfall['high'] = bell_mf(rainfall.universe, 10, 2, 70)
rainfall['very_high'] = bell_mf(rainfall.universe, 10, 2, 85)
rainfall['extremely_high'] = bell_mf(rainfall.universe, 8, 2, 100)

# Define humidity membership functions
humidity['very_low'] = bell_mf(humidity.universe, 8, 2, 5)
humidity['low'] = bell_mf(humidity.universe, 10, 2, 20)
humidity['mid'] = bell_mf(humidity.universe, 10, 2, 40)
humidity['high'] = bell_mf(humidity.universe, 10, 2, 60)
humidity['very_high'] = bell_mf(humidity.universe, 10, 2, 80)
humidity['extremely_high'] = bell_mf(humidity.universe, 8, 2, 95)

# Define temperature membership functions
temperature['low'] = bell_mf(temperature.universe, 4, 2, 10)
temperature['mid'] = bell_mf(temperature.universe, 5, 2, 20)
temperature['high'] = bell_mf(temperature.universe, 5, 2, 30)
temperature['very_high'] = bell_mf(temperature.universe, 5, 2, 40)
temperature['extremely_high'] = bell_mf(temperature.universe, 4, 2, 50)

# Define pH membership functions
ph['low'] = bell_mf(ph.universe, 1, 2, 4)
ph['high'] = bell_mf(ph.universe, 1, 2, 8)

# Output membership functions (TSK 0-order crisp values)
pump['OFF'] = fuzz.trimf(pump.universe, [0, 0, 0])
pump['ON'] = fuzz.trimf(pump.universe, [1, 1, 1])

# Define rules (add or customize based on your logic)
rules = [
    ctrl.Rule(soil_moisture['very_high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['very_low'] & humidity['very_high'] & temperature['high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['very_low'] & humidity['very_high'] & temperature['mid'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['very_low'] & humidity['mid'] & ph['high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['very_low'] & humidity['extremely_high'] & temperature['very_high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['very_low'] & humidity['extremely_high'] & temperature['high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['very_low'] & humidity['extremely_high'] & temperature['mid'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['very_low'] & humidity['high'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['very_low'] & humidity['low'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['high'] & humidity['very_low'] & temperature['low'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['high'] & humidity['extremely_high'] & temperature['very_high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['high'] & humidity['extremely_high'] & temperature['high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['high'] & humidity['extremely_high'] & temperature['mid'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['high'] & humidity['mid'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['high'] & humidity['very_high'] & temperature['mid'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['high'] & humidity['very_high'] & temperature['high'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['high'] & humidity['low'] & temperature['mid'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['high'] & humidity['low'] & temperature['very_high'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['high'] & humidity['low'] & temperature['high'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['high'] & humidity['low'] & temperature['low'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['high'] & humidity['high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['low'] & humidity['extremely_high'] & temperature['very_high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['low'] & humidity['extremely_high'] & temperature['high'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['low'] & humidity['extremely_high'] & temperature['mid'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['low'] & humidity['mid'] & ph['low'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['low'] & humidity['mid'] & ph['high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['low'] & humidity['high'] & ph['low'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['low'] & humidity['high'] & ph['high'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['low'] & humidity['very_high'] & temperature['low'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['low'] & humidity['very_high'] & temperature['mid'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['low'] & humidity['very_high'] & temperature['very_high'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['low'] & humidity['very_high'] & temperature['very_high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['low'] & humidity['very_low'] & temperature['low'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['low'] & humidity['very_low'] & temperature['mid'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['low'] & humidity['low'] & temperature['high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['low'] & humidity['low'] & temperature['mid'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['mid'] & humidity['extremely_high'] & temperature['low'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['mid'] & humidity['extremely_high'] & temperature['mid'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['mid'] & humidity['extremely_high'] & temperature['high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['mid'] & humidity['extremely_high'] & temperature['very_high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['mid'] & humidity['very_low'] & ph['low'] , pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['mid'] & humidity['very_low'] & ph['high'] , pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['mid'] & humidity['low'] & temperature['high'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['mid'] & humidity['low'] & temperature['mid'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['mid'] & humidity['low'] & temperature['very_high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['mid'] & humidity['very_high'] & ph['low'] , pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['mid'] & humidity['very_high'] & ph['high'] , pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['mid'] & humidity['mid'] & temperature['mid'] , pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['mid'] & humidity['mid'] & temperature['low'] , pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['mid'] & humidity['high'] & temperature['mid'] , pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['very_high'] & humidity['high'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['very_high'] & humidity['low'] & temperature['high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['very_high'] & humidity['low'] & temperature['low'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['very_high'] & humidity['extremely_high'] & temperature['high'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['very_high'] & humidity['extremely_high'] & temperature['mid'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['very_high'] & humidity['extremely_high'] & temperature['very_high'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['very_high'] & humidity['very_high'] & ph['low'] , pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['very_high'] & humidity['very_high'] & ph['high'] , pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['very_high'] & humidity['mid'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['extremely_low'] & humidity['low'] & temperature['high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['extremely_low'] & humidity['extremely_high'] & temperature['high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['extremely_low'] & humidity['mid'] & temperature['high'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['extremely_low'] & temperature['mid'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['extremely_low'] & temperature['low'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['extremely_high'] & temperature['mid'] & ph['low'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['extremely_high'] & temperature['mid'] & ph['high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['extremely_high'] & temperature['low'], pump['ON']),
    ctrl.Rule(soil_moisture['high'] & rainfall['extremely_high'] & humidity['extremely_high'] & temperature['high'], pump['OFF']),
    ctrl.Rule(soil_moisture['high'] & rainfall['extremely_high'] & humidity['very_high'] & temperature['high'], pump['OFF']),
    ctrl.Rule(soil_moisture['low'], pump['ON']),
    ctrl.Rule(soil_moisture['medium'], pump['ON'])
]

# Create control system
pump_ctrl = ctrl.ControlSystem(rules)
pump_sim = ctrl.ControlSystemSimulation(pump_ctrl)

# Provide input values
pump_sim.input['soil_moisture'] = 45
pump_sim.input['rainfall'] = 30
pump_sim.input['humidity'] = 60
pump_sim.input['temperature'] = 25
pump_sim.input['ph'] = 7

# Compute output
pump_sim.compute()

# Display results
output1 = pump_sim.output['pump']
print("Pump activation level (0=OFF, 1=ON):", round(output1, 2))

# Apply threshold
if output1 >= 0.5:
    print("Pump is ON")
else:
    print("Pump is OFF")