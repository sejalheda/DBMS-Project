import pandas as pd
import joblib

model = joblib.load("priority_model.pkl")

new_complaint = pd.DataFrame([
    {
        "description": "Large garbage pile outside hospital",
        "category": "Garbage",
        "location_type": "Hospital",
        "duplicate_count": 3
    }
])

prediction = model.predict(new_complaint)

print("Predicted Priority:", int(prediction[0]))