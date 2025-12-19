from flask import Flask, render_template
from datetime import datetime

app = Flask(__name__)

# Shared fall data (updated by detection script)
fall_data = {
    "velocity": 0.0,
    "time": "",
    "severity": "UNKNOWN",
    "status": "Monitoring"
}

@app.route("/")
def alert():
    return render_template(
        "alert.html",
        velocity=fall_data["velocity"],
        time=fall_data["time"],
        severity=fall_data["severity"],
        status=fall_data["status"]
    )

def update_fall_data(velocity):
    fall_data["velocity"] = velocity
    fall_data["time"] = datetime.now().strftime("%d %b %Y, %H:%M:%S")

    # Severity logic (simple but effective)
    if velocity < 1.2:
        fall_data["severity"] = "LOW"
    elif velocity < 2.5:
        fall_data["severity"] = "MODERATE"
    else:
        fall_data["severity"] = "HIGH"

    fall_data["status"] = "🚨 FALL CONFIRMED"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

