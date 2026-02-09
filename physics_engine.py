import numpy as np

class PhysicsEngine:
    def __init__(self, specific_gravity=0.85):
        self.sg = specific_gravity
        
    def calculate_tdh(self, pip, discharge_press):
        """
        Calculate Total Dynamic Head (TDH) in feet.
        TDH = (Discharge - Intake) * 2.31 / SG
        """
        return (discharge_press - pip) * 2.31 / self.sg

    def get_ideal_head(self, flow_rate):
        """
        Simplified Manufacturer Pump Curve: H = A - B*Q^2
        Example: H = 5000 - 0.001 * Q^2
        """
        # Coefficients for a representative ESP pump curve
        A = 5000
        B = 0.001
        return A - B * (flow_rate**2)

    def check_health(self, pip, discharge_press, flow_rate):
        """
        Calculates degradation based on the gap between ideal and actual TDH.
        """
        actual_tdh = self.calculate_tdh(pip, discharge_press)
        ideal_head = self.get_ideal_head(flow_rate)
        
        # Avoid division by zero
        if ideal_head == 0:
            return 0.0
            
        degradation = (ideal_head - actual_tdh) / ideal_head
        return degradation

    def get_optimization_advice(self, pip, amps, degradation):
        """
        Provides advisory based on physics and sensor triggers.
        """
        advice = []
        
        # Gas Lock Check (Simplified)
        if pip < 350 and amps > 65:
            advice.append("Potential Gas Lock detected. Suggest increasing Hz to 55-60 to clear gas.")
        
        # Mechanical Degradation Check
        if degradation > 0.15:
            advice.append("Significant pump degradation detected (>15%). Schedule maintenance check.")
            
        if not advice:
            advice.append("System operating within normal physics parameters.")
            
        return " | ".join(advice)

if __name__ == "__main__":
    engine = PhysicsEngine()
    # Test case
    pip = 400
    discharge = 2400
    flow = 1000
    
    tdh = engine.calculate_tdh(pip, discharge)
    ideal = engine.get_ideal_head(flow)
    deg = engine.check_health(pip, discharge, flow)
    
    print(f"Actual TDH: {tdh:.2f} ft")
    print(f"Ideal Head: {ideal:.2f} ft")
    print(f"Degradation: {deg*100:.2f}%")
