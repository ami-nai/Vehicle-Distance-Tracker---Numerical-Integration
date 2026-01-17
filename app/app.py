from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

def trapezoidal_rule(time, speed):
    cumulative = np.zeros(len(time))
    for i in range(1, len(time)):
        h = time[i] - time[i-1]
        cumulative[i] = cumulative[i-1] + (h/2)*(speed[i] + speed[i-1])
    return cumulative[-1], cumulative

def simpsons_rule(time, speed):
    cumulative = np.zeros(len(time))
    for i in range(2, len(time), 2):
        h = time[i] - time[i-2]
        cumulative[i] = cumulative[i-2] + (h/6)*(speed[i-2] + 4*speed[i-1] + speed[i])
        cumulative[i-1] = (cumulative[i-2] + cumulative[i]) / 2
    if len(time) % 2 == 0:
        h = time[-1] - time[-2]
        cumulative[-1] = cumulative[-2] + (h/2)*(speed[-1] + speed[-2])
    return cumulative[-1], cumulative

def actual_distance(time, speed):
    """
    Calculate actual distance using rectangle method.
    Distance_i = Speed_i × Δt_i
    Total Distance = Σ(Speed_i × Δt_i)
    """
    cumulative = np.zeros(len(time))
    
    for i in range(1, len(time)):
        delta_t = time[i] - time[i-1]
        distance_segment = speed[i] * delta_t
        cumulative[i] = cumulative[i-1] + distance_segment
    
    return cumulative[-1], cumulative

def plot_results(time, speed, dist_actual, dist_trap, dist_simp):
    fig = plt.figure(figsize=(10, 10))
    
    # Speed vs Time
    plt.subplot(3, 1, 1)
    plt.plot(time, speed, marker='o', color='#3b82f6', linewidth=2, markersize=5)
    plt.fill_between(time, speed, alpha=0.3, color='#3b82f6')
    plt.title("Speed vs Time", fontsize=12, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.grid(alpha=0.3)

    # Cumulative Distance Comparison
    plt.subplot(3, 1, 2)
    plt.plot(time, dist_actual, label="Actual (Rectangle Method)", linewidth=3, color='#000000', linestyle='-', alpha=0.7)
    plt.plot(time, dist_trap, label="Trapezoidal Rule", marker='s', linewidth=2, color='#ef4444', markersize=4)
    plt.plot(time, dist_simp, label="Simpson's Rule", marker='^', linewidth=2, color='#22c55e', markersize=4)
    plt.title("Cumulative Distance Comparison", fontsize=12, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.legend()
    plt.grid(alpha=0.3)

    # Error Analysis
    plt.subplot(3, 1, 3)
    error_trap = np.abs(dist_actual - dist_trap)
    error_simp = np.abs(dist_actual - dist_simp)
    
    plt.plot(time, error_trap, label="Trapezoidal Error", marker='s', linewidth=2, color='#ef4444', markersize=4)
    plt.plot(time, error_simp, label="Simpson's Error", marker='^', linewidth=2, color='#22c55e', markersize=4)
    plt.title("Absolute Error vs Actual Distance", fontsize=12, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Absolute Error (m)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("static/result.png", dpi=150, bbox_inches='tight')
    plt.close()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["file"]
        df = pd.read_csv(file)

        time = df["Time (s)"].values
        speed = df["Speed (m/s)"].values

        # Calculate actual distance using rectangle method
        actual_total, actual_dist = actual_distance(time, speed)
        
        # Calculate distances using numerical methods
        trap_total, trap_dist = trapezoidal_rule(time, speed)
        simp_total, simp_dist = simpsons_rule(time, speed)

        # Calculate final errors
        trap_error = abs(actual_total - trap_total)
        simp_error = abs(actual_total - simp_total)
        trap_rel_error = (trap_error / actual_total) * 100 if actual_total != 0 else 0
        simp_rel_error = (simp_error / actual_total) * 100 if actual_total != 0 else 0

        plot_results(time, speed, actual_dist, trap_dist, simp_dist)

        result = {
            "actual": round(actual_total, 4),
            "trap": round(trap_total, 4),
            "simp": round(simp_total, 4),
            "trap_error": round(trap_error, 4),
            "simp_error": round(simp_error, 4),
            "trap_rel_error": round(trap_rel_error, 4),
            "simp_rel_error": round(simp_rel_error, 4)
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
