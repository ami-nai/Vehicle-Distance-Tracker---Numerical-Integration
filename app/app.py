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

def plot_results(time, speed, dist_trap, dist_simp):
    plt.figure(figsize=(8,6))
    plt.subplot(2,1,1)
    plt.plot(time, speed, marker='o')
    plt.title("Speed vs Time")
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(time, dist_trap, label="Trapezoidal")
    plt.plot(time, dist_simp, label="Simpson")
    plt.title("Cumulative Distance")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("static/result.png")
    plt.close()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["file"]
        df = pd.read_csv(file)

        time = df["Time (s)"].values
        speed = df["Speed (m/s)"].values

        trap_total, trap_dist = trapezoidal_rule(time, speed)
        simp_total, simp_dist = simpsons_rule(time, speed)

        plot_results(time, speed, trap_dist, simp_dist)

        result = {
            "trap": round(trap_total, 2),
            "simp": round(simp_total, 2),
            "diff": round(abs(trap_total - simp_total), 2)
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
