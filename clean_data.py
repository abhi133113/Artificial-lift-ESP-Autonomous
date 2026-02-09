"""
Data Cleaning Pipeline for ESP Pump Sensor Data

This script performs comprehensive data cleaning:
1. Handle missing values
2. Remove outliers using IQR method
3. Smooth noisy signals
4. Feature engineering (add rolling statistics)
5. Validate data quality
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
import os

class ESPDataCleaner:
    def __init__(self):
        self.features = ['PIP', 'Discharge_Press', 'Amps', 'Temp', 'Vibration']
        self.stats = {}
        
    def clean(self, df, verbose=True):
        """Main cleaning pipeline."""
        if verbose:
            print("="*60)
            print("ESP DATA CLEANING PIPELINE")
            print("="*60)
            print(f"\nOriginal data shape: {df.shape}")
        
        df_clean = df.copy()
        
        # Step 1: Handle missing values
        df_clean = self._handle_missing(df_clean, verbose)
        
        # Step 2: Remove outliers
        df_clean = self._remove_outliers(df_clean, verbose)
        
        # Step 3: Smooth noisy signals
        df_clean = self._smooth_signals(df_clean, verbose)
        
        # Step 4: Add rolling features
        df_clean = self._add_rolling_features(df_clean, verbose)
        
        # Step 5: Validate
        self._validate(df_clean, verbose)
        
        if verbose:
            print(f"\nFinal cleaned data shape: {df_clean.shape}")
            print("="*60)
        
        return df_clean
    
    def _handle_missing(self, df, verbose):
        """Handle missing values with forward fill then backward fill."""
        missing_before = df[self.features].isnull().sum().sum()
        
        # Forward fill, then backward fill
        df[self.features] = df[self.features].ffill().bfill()
        
        # If still missing, fill with column mean
        for col in self.features:
            if df[col].isnull().any():
                df[col].fillna(df[col].mean(), inplace=True)
        
        missing_after = df[self.features].isnull().sum().sum()
        
        if verbose:
            print(f"\n[1] Missing Values: {missing_before} → {missing_after}")
        
        return df
    
    def _remove_outliers(self, df, verbose):
        """Remove outliers using IQR method with 3x IQR bounds."""
        outliers_removed = 0
        
        for col in self.features:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Cap outliers instead of removing rows
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outliers_removed += outliers
            
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            self.stats[col] = {
                'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
                'lower': lower_bound, 'upper': upper_bound
            }
        
        if verbose:
            print(f"[2] Outliers Capped: {outliers_removed} values")
        
        return df
    
    def _smooth_signals(self, df, verbose):
        """Apply Savitzky-Golay filter to smooth noisy signals."""
        # Only smooth vibration (typically the noisiest)
        if 'Vibration' in df.columns and len(df) > 11:
            df['Vibration'] = savgol_filter(df['Vibration'], window_length=11, polyorder=3)
        
        if verbose:
            print("[3] Signals Smoothed: Savitzky-Golay filter applied to Vibration")
        
        return df
    
    def _add_rolling_features(self, df, verbose):
        """Add rolling mean and std for better anomaly detection."""
        window = 10
        
        for col in ['PIP', 'Amps', 'Vibration']:
            if col in df.columns:
                df[f'{col}_rolling_mean'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_rolling_std'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
        
        # Add rate of change for key features
        for col in ['PIP', 'Amps']:
            if col in df.columns:
                df[f'{col}_rate'] = df[col].diff().fillna(0)
        
        if verbose:
            print(f"[4] Rolling Features Added: mean, std (window={window}), rate of change")
        
        return df
    
    def _validate(self, df, verbose):
        """Validate cleaned data quality."""
        issues = []
        
        # Check for any remaining NaN
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            issues.append(f"  - {nan_count} NaN values remaining")
        
        # Check for infinite values
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            issues.append(f"  - {inf_count} Inf values found")
        
        # Check for constant columns
        for col in self.features:
            if df[col].std() == 0:
                issues.append(f"  - {col} has zero variance")
        
        if verbose:
            if issues:
                print("[5] Validation Issues:")
                for issue in issues:
                    print(issue)
            else:
                print("[5] Validation: ✓ All checks passed")

def main():
    print("\n" + "="*60)
    print("CLEANING REAL PUMP SENSOR DATA")
    print("="*60)
    
    # Load raw data
    df = pd.read_csv('data/real_pump_esp_format.csv')
    print(f"\nLoaded {len(df)} samples")
    
    # Show data statistics before cleaning
    print("\nData Statistics BEFORE Cleaning:")
    print(df[['PIP', 'Discharge_Press', 'Amps', 'Temp', 'Vibration']].describe())
    
    # Clean data
    cleaner = ESPDataCleaner()
    df_clean = cleaner.clean(df)
    
    # Show data statistics after cleaning
    print("\nData Statistics AFTER Cleaning:")
    features = ['PIP', 'Discharge_Press', 'Amps', 'Temp', 'Vibration']
    print(df_clean[features].describe())
    
    # Save cleaned data
    df_clean.to_csv('data/real_pump_cleaned.csv', index=False)
    print(f"\nSaved cleaned data to data/real_pump_cleaned.csv")
    print(f"New features added: {[c for c in df_clean.columns if c not in df.columns]}")

if __name__ == "__main__":
    main()
