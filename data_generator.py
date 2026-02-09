import numpy as np
import pandas as pd
import os

def generate_esp_data(n_samples=2000, failure_mode=None):
    """
    Generates synthetic ESP sensor data.
    
    Parameters:
    - n_samples: Number of data points.
    - failure_mode: 'gas_lock', 'pump_degradation', or None for normal.
    """
    t = np.linspace(0, 100, n_samples)
    
    # Base Normal Parameters
    pip = 500 + np.random.normal(0, 5, n_samples) # Intake Pressure (PSI)
    discharge_press = 2500 + np.random.normal(0, 10, n_samples) # Discharge Pressure (PSI)
    motor_amps = 60 + np.random.normal(0, 1, n_samples) # Motor Amps
    motor_temp = 150 + np.random.normal(0, 2, n_samples) # Motor Temp (F)
    vibration = 0.5 + np.random.normal(0, 0.05, n_samples) # Vibration (in/s)
    flow_rate = 1000 + np.random.normal(0, 20, n_samples) # Flow Rate (BPD)
    hz = np.full(n_samples, 60.0) # Frequency (Hz)

    if failure_mode == 'gas_lock':
        # Gas Lock: PIP drops, Amps fluctuate, Flow drops
        start_idx = int(n_samples * 0.7)
        pip[start_idx:] -= np.linspace(0, 200, n_samples - start_idx)
        motor_amps[start_idx:] += np.sin(np.arange(n_samples - start_idx) * 0.5) * 8
        flow_rate[start_idx:] -= np.linspace(0, 400, n_samples - start_idx)
        vibration[start_idx:] += np.random.normal(0, 0.2, n_samples - start_idx)
        
    elif failure_mode == 'pump_degradation':
        # Pump Degradation: Gradual increase in vibration, slight drop in efficiency (Head/Flow)
        start_idx = int(n_samples * 0.5)
        vibration[start_idx:] += np.linspace(0, 1.5, n_samples - start_idx)
        discharge_press[start_idx:] -= np.linspace(0, 300, n_samples - start_idx)
        motor_temp[start_idx:] += np.linspace(0, 30, n_samples - start_idx)

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='min'),
        'PIP': pip,
        'Discharge_Press': discharge_press,
        'Amps': motor_amps,
        'Temp': motor_temp,
        'Vibration': vibration,
        'Flow_Rate': flow_rate,
        'Hz': hz
    })
    
    return df

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate Normal Data
    normal_df = generate_esp_data(n_samples=5000)
    normal_df.to_csv('data/esp_normal.csv', index=False)
    print("Generated data/esp_normal.csv")
    
    # Generate Gas Lock Data
    gas_lock_df = generate_esp_data(n_samples=2000, failure_mode='gas_lock')
    gas_lock_df.to_csv('data/esp_gas_lock.csv', index=False)
    print("Generated data/esp_gas_lock.csv")
    
    # Generate Degradation Data
    degradation_df = generate_esp_data(n_samples=2000, failure_mode='pump_degradation')
    degradation_df.to_csv('data/esp_degradation.csv', index=False)
    print("Generated data/esp_degradation.csv")
