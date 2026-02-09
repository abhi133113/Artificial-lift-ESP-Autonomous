import numpy as np
import pandas as pd
import os

def generate_user_data():
    """
    Generates data based strictly on the user's provided snippet.
    Fills missing columns with normal noise to match model input shape.
    """
    # User Snippet Logic
    t = np.linspace(0, 100, 1000)
    intake_press = 500 + np.random.normal(0, 5, 1000) # PSI
    motor_amps = 60 + np.random.normal(0, 1, 1000)    # Amps

    # Simulate a "Gas Lock" event (Pressure drops, Amps become unstable)
    # User: intake_press[800:] = intake_press[800:] - (np.arange(200) * 1.5)
    # User: motor_amps[800:] = motor_amps[800:] + np.sin(np.arange(200)) * 5
    
    intake_press[800:] = intake_press[800:] - (np.arange(200) * 1.5)
    motor_amps[800:] = motor_amps[800:] + np.sin(np.arange(200)) * 5

    # Fill other columns with "Normal" behavior
    # Discharge Pressure: 2500 +/- 10
    discharge_press = 2500 + np.random.normal(0, 10, 1000)
    # Temp: 150 +/- 2
    motor_temp = 150 + np.random.normal(0, 2, 1000)
    # Vibration: 0.5 +/- 0.05
    vibration = 0.5 + np.random.normal(0, 0.05, 1000)
    
    # Note: The user snippet implies Gas Lock. 
    # In a real gas lock, vibration might increase, but the user didn't specify it.
    # We will keep vibration normal to test if the model can detect it purely on PIP/Amps.

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='min'),
        'PIP': intake_press,
        'Discharge_Press': discharge_press,
        'Amps': motor_amps,
        'Temp': motor_temp,
        'Vibration': vibration,
        'Flow_Rate': 1000, # Placeholder
        'Hz': 60.0
    })
    
    return df

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    df = generate_user_data()
    df.to_csv('data/user_gas_lock.csv', index=False)
    print("Generated data/user_gas_lock.csv based on user snippet.")
