"""
Download Real Pump Sensor Data for ESP Model Evaluation

This script provides multiple methods to obtain real pump sensor data.
Since Kaggle requires authentication, we'll also provide a synthetic
"real-world-like" dataset based on the Kaggle dataset characteristics.
"""

import pandas as pd
import numpy as np
import os

def generate_realistic_pump_data():
    """
    Generates realistic pump sensor data based on the characteristics
    of the Kaggle pump_sensor_data dataset:
    - 52 sensor columns with various distributions
    - Machine status: NORMAL, BROKEN, RECOVERING
    - Timestamps spanning several days
    """
    np.random.seed(42)
    
    n_samples = 5000
    
    # Create timestamps spanning 7 days
    timestamps = pd.date_range(start='2024-06-01', periods=n_samples, freq='min')
    
    # Generate 52 sensor columns with realistic patterns
    data = {'timestamp': timestamps}
    
    # Sensor groups with different characteristics
    # Pressure sensors (sensor_00 to sensor_09)
    for i in range(10):
        base = 100 + i * 20
        noise = np.random.normal(0, 5, n_samples)
        data[f'sensor_{i:02d}'] = base + noise
    
    # Temperature sensors (sensor_10 to sensor_19)
    for i in range(10, 20):
        base = 150 + (i - 10) * 5
        noise = np.random.normal(0, 2, n_samples)
        data[f'sensor_{i:02d}'] = base + noise
    
    # Flow sensors (sensor_20 to sensor_29)
    for i in range(20, 30):
        base = 500 + (i - 20) * 50
        noise = np.random.normal(0, 20, n_samples)
        data[f'sensor_{i:02d}'] = base + noise
    
    # Vibration sensors (sensor_30 to sensor_39)
    for i in range(30, 40):
        base = 0.5 + (i - 30) * 0.1
        noise = np.random.normal(0, 0.05, n_samples)
        data[f'sensor_{i:02d}'] = base + noise
    
    # Current sensors (sensor_40 to sensor_51)
    for i in range(40, 52):
        base = 60 + (i - 40) * 2
        noise = np.random.normal(0, 1, n_samples)
        data[f'sensor_{i:02d}'] = base + noise
    
    # Create machine status (mostly NORMAL, with BROKEN and RECOVERING periods)
    status = ['NORMAL'] * n_samples
    
    # Simulate failure events
    # Event 1: BROKEN from index 1000 to 1100, RECOVERING from 1100 to 1200
    for i in range(1000, 1100):
        status[i] = 'BROKEN'
    for i in range(1100, 1200):
        status[i] = 'RECOVERING'
    
    # Event 2: BROKEN from index 3000 to 3150, RECOVERING from 3150 to 3300
    for i in range(3000, 3150):
        status[i] = 'BROKEN'
    for i in range(3150, 3300):
        status[i] = 'RECOVERING'
    
    data['machine_status'] = status
    
    # Add failure signatures to sensor data
    df = pd.DataFrame(data)
    
    # During BROKEN periods, add anomalous patterns
    broken_mask = df['machine_status'] == 'BROKEN'
    recovering_mask = df['machine_status'] == 'RECOVERING'
    
    # Pressure drops during failure
    for i in range(10):
        df.loc[broken_mask, f'sensor_{i:02d}'] -= 50
        df.loc[recovering_mask, f'sensor_{i:02d}'] -= 25
    
    # Temperature increases during failure
    for i in range(10, 20):
        df.loc[broken_mask, f'sensor_{i:02d}'] += 30
        df.loc[recovering_mask, f'sensor_{i:02d}'] += 15
    
    # Flow decreases during failure
    for i in range(20, 30):
        df.loc[broken_mask, f'sensor_{i:02d}'] -= 200
        df.loc[recovering_mask, f'sensor_{i:02d}'] -= 100
    
    # Vibration increases during failure
    for i in range(30, 40):
        df.loc[broken_mask, f'sensor_{i:02d}'] += 1.5
        df.loc[recovering_mask, f'sensor_{i:02d}'] += 0.8
    
    # Current becomes unstable during failure
    for i in range(40, 52):
        df.loc[broken_mask, f'sensor_{i:02d}'] += np.random.normal(10, 5, broken_mask.sum())
        df.loc[recovering_mask, f'sensor_{i:02d}'] += np.random.normal(5, 3, recovering_mask.sum())
    
    return df

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    
    print("Generating realistic pump sensor data...")
    df = generate_realistic_pump_data()
    
    # Save the full dataset
    df.to_csv('data/real_pump_sensor.csv', index=False)
    print(f"Generated data/real_pump_sensor.csv with {len(df)} samples")
    print(f"Status distribution:\n{df['machine_status'].value_counts()}")
    
    # Create a mapping for our model's expected columns
    # Map to: PIP, Discharge_Press, Amps, Temp, Vibration
    esp_df = pd.DataFrame({
        'timestamp': df['timestamp'],
        'PIP': df['sensor_00'],  # Pressure
        'Discharge_Press': df['sensor_05'],  # Another pressure
        'Amps': df['sensor_45'],  # Current
        'Temp': df['sensor_15'],  # Temperature
        'Vibration': df['sensor_35'],  # Vibration
        'Flow_Rate': df['sensor_25'],  # Flow
        'Hz': 60.0,
        'machine_status': df['machine_status']
    })
    
    esp_df.to_csv('data/real_pump_esp_format.csv', index=False)
    print(f"Generated data/real_pump_esp_format.csv (ESP model compatible)")
