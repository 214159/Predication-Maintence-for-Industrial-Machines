from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# --- LOAD MODELS ---
# Ensure these files are in your main project folder
clf = joblib.load('classifier_model.pkl')
reg = joblib.load('regressor_model.pkl')
le = joblib.load('label_encoder.pkl')

def get_recovery_advice(feature_names):
    """Provides human-readable advice based on technical feature impacts"""
    advice_db = {
        'Tool_wear_min': "⚠️ Tool usage is nearing limit. Schedule a replacement to prevent snapping.",
        'Torque_Nm': "⚙️ High torque detected. Inspect motor load and check for material hardness issues.",
        'Rotational_speed_rpm': "🌀 RPM instability. Check spindle bearings and belt tension.",
        'Air_temperature_K': "🌡️ Factory floor is too hot. Check ventilation or AC units.",
        'Process_temperature_K': "🔥 Excessive process heat. Increase coolant flow or reduce feed rate."
    }
    return [advice_db.get(f, f"🔍 Monitor {f} for further anomalies.") for f in feature_names]

def encode_machine_quality(quality_input):
    """Maps 1 selection to the 3 binary columns: [Type_H, Type_L, Type_M]"""
    q = str(quality_input).strip().capitalize()
    if q == 'High': return [1, 0, 0]
    elif q == 'Low': return [0, 1, 0]
    else: return [0, 0, 1] # Default to Medium

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Handle JSON (from CMD/test.py) or Form (from Website)
        data = request.get_json() if request.is_json else request.form

        # 1. Capture 6 Neat Inputs
        q_choice = data['machine_quality']
        air_t = float(data['air_t'])
        proc_t = float(data['proc_t'])
        rpm = float(data['rpm'])
        torque = float(data['torque'])
        wear = float(data['wear'])

        # 2. Convert to 8 Technical Features
        # Order: [Type_H, Type_L, Type_M, Air_temp, Proc_temp, RPM, Torque, Wear]
        encoded_type = encode_machine_quality(q_choice)
        sensor_data = [air_t, proc_t, rpm, torque, wear]
        final_features = np.array([encoded_type + sensor_data])

        # 3. Model Logic
        # Classification (Is it broken?)
        pred_idx = clf.predict(final_features)[0]
        status = le.inverse_transform([pred_idx])[0]
        
        # Regression (When will it break?)
        time_val = round(float(reg.predict(final_features)[0]), 2)

        # 4. Impact Analysis (Which sensor is causing the issue?)
        # Using exact column names from your dataset for the logic
        feat_names = ['Type_H', 'Type_L', 'Type_M', 'Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
        importances = clf.feature_importances_
        top_indices = np.argsort(importances)[::-1][:2]
        top_2_feats = [feat_names[i] for i in top_indices]
        
        # Clean up names for the UI display
        ui_impacts = [f.replace('_', ' ').replace('K', '').replace('Nm', '').replace('min', '').strip() for f in top_2_feats]
        recovery = get_recovery_advice(top_2_feats)

        res = {
            "status": status,
            "time": "System Healthy" if status == "No_Failure" else f"{time_val} Min",
            "estimated_minutes": time_val, 
            "top_impacts": ui_impacts,
            "recovery_steps": recovery
        }

        return jsonify(res)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)