# src/recommend.py

# Script to generate recommendations based on predicted supply-demand balance.
# Uses predefined bands to classify predictions and provide actions for grid operators and customers alike based on the classification band.

def band_label(pred, sigma, neutral_k=0.25, high_k=0.75):
    if pred >= high_k * sigma: return "HIGH SURPLUS"
    if pred >= neutral_k * sigma: return "SURPLUS"
    if pred <= -high_k * sigma: return "HIGH DEFICIT"
    if pred <= -neutral_k * sigma: return "DEFICIT"
    return "NEUTRAL"

def recommendation_from_band(band):
    mapping = {
        "HIGH SURPLUS": {
            "description": "There is an excess of generation over load. Electricity price signals may negative.",
            "grid_action": "Charge batteries and schedule storage to absorb surplus. Export if possible.",
            "customer_action": "Encourage EV charging and smart appliances now for lower rates."
        },
        "SURPLUS": {
            "description": "There is a moderate excess of generation over load. Electricity price signals may be low.",
            "grid_action": "Opportunistic charging; reduce reserve procurement.",
            "customer_action": "Shift flexible loads to this window. Consider EV charging or running appliances."
        },
        "NEUTRAL": {
            "description": "Generation and load are balanced. Electricity price signals are stable.",
            "grid_action": "No large actions recommended. System is stable.",
            "customer_action": "No special action. Normal operations."
        },
        "DEFICIT": {
            "description": "There is a moderate excess of load over generation. Electricity price signals may be high.",
            "grid_action": "Moderate discharge or demand response (DR) as needed.",
            "customer_action": "Avoid adding flexible load, as rates may be higher."
        },
        "HIGH DEFICIT": {
            "description": "There is a critical excess of load over generation. Electricity price signals may be very high.",
            "grid_action": "Immediate action required. Discharge storage, engage DR programs, or increase imports.",
            "customer_action": "Defer EV charging and non-essential loads to avoid high rates and support grid stability."
        }
    }
    return mapping.get(band, mapping["NEUTRAL"])