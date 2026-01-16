"""
Vehicle Distance Tracker
Uses numerical integration (Trapezoidal and Simpson's Rule) to calculate 
total distance traveled from speed-time data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple


def trapezoidal_rule(time: np.ndarray, speed: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Calculate distance using Trapezoidal Rule.
    
    Args:
        time: Array of time values (s)
        speed: Array of speed values (m/s)
    
    Returns:
        total_distance: Total distance traveled (m)
        cumulative_distance: Array of cumulative distances at each time point
    """
    n = len(time)
    cumulative_distance = np.zeros(n)
    
    for i in range(1, n):
        h = time[i] - time[i-1]
        # Trapezoidal rule: distance = (h/2) * (v_i + v_{i-1})
        distance_segment = (h / 2) * (speed[i] + speed[i-1])
        cumulative_distance[i] = cumulative_distance[i-1] + distance_segment
    
    total_distance = cumulative_distance[-1]
    return total_distance, cumulative_distance


def simpsons_rule(time: np.ndarray, speed: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Calculate distance using Simpson's 1/3 Rule.
    
    Args:
        time: Array of time values (s)
        speed: Array of speed values (m/s)
    
    Returns:
        total_distance: Total distance traveled (m)
        cumulative_distance: Array of cumulative distances at each time point
    """
    n = len(time)
    cumulative_distance = np.zeros(n)
    
    for i in range(2, n, 2):
        h = time[i] - time[i-2]
        # Simpson's 1/3 rule: distance = (h/6) * (v_i-2 + 4*v_i-1 + v_i)
        distance_segment = (h / 6) * (speed[i-2] + 4*speed[i-1] + speed[i])
        cumulative_distance[i] = cumulative_distance[i-2] + distance_segment
        # Linear interpolation for odd index points
        if i > 2:
            cumulative_distance[i-1] = (cumulative_distance[i-2] + cumulative_distance[i]) / 2
    
    # Handle case where n is even (last point already calculated)
    # For odd n, use trapezoidal for the last segment
    if n % 2 == 0:
        h = time[-1] - time[-2]
        distance_segment = (h / 2) * (speed[-1] + speed[-2])
        cumulative_distance[-1] = cumulative_distance[-2] + distance_segment
    
    total_distance = cumulative_distance[-1]
    return total_distance, cumulative_distance


def load_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load speed-time data from CSV file.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        time: Array of time values
        speed: Array of speed values
    """
    df = pd.read_csv(filepath)
    time = df['Time (s)'].values
    speed = df['Speed (m/s)'].values
    return time, speed


def plot_results(time: np.ndarray, speed: np.ndarray, 
                 dist_trap: np.ndarray, dist_simp: np.ndarray,
                 total_trap: float, total_simp: float):
    """
    Visualize speed vs time and cumulative distance vs time.
    
    Args:
        time: Array of time values
        speed: Array of speed values
        dist_trap: Cumulative distance using Trapezoidal Rule
        dist_simp: Cumulative distance using Simpson's Rule
        total_trap: Total distance (Trapezoidal)
        total_simp: Total distance (Simpson's)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Speed vs Time
    ax1.plot(time, speed, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Speed (m/s)', fontsize=12)
    ax1.set_title('Speed vs Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(time, speed, alpha=0.3)
    
    # Cumulative Distance vs Time
    ax2.plot(time, dist_trap, 'r-', linewidth=2, label='Trapezoidal Rule', marker='s', markersize=4)
    ax2.plot(time, dist_simp, 'g--', linewidth=2, label="Simpson's Rule", marker='^', markersize=4)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Cumulative Distance (m)', fontsize=12)
    ax2.set_title('Cumulative Distance vs Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add text box with results
    textstr = f'Total Distance:\nTrapezoidal: {total_trap:.2f} m\nSimpson\'s: {total_simp:.2f} m\nDifference: {abs(total_trap - total_simp):.2f} m'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('vehicle_distance_tracker_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to run the Vehicle Distance Tracker."""
    print("=" * 60)
    print("Vehicle Distance Tracker - Numerical Integration")
    print("=" * 60)
    
    # Load data
    filepath = "data/speed_time_data.csv"
    print(f"\nLoading data from {filepath}...")
    time, speed = load_data(filepath)
    print(f"Loaded {len(time)} data points")
    print(f"Time range: {time[0]:.1f} - {time[-1]:.1f} seconds")
    print(f"Speed range: {speed.min():.1f} - {speed.max():.1f} m/s")
    
    # Calculate distance using Trapezoidal Rule
    print("\n" + "-" * 60)
    print("Calculating distance using Trapezoidal Rule...")
    total_distance_trap, cumulative_distance_trap = trapezoidal_rule(time, speed)
    print(f"Total Distance (Trapezoidal): {total_distance_trap:.2f} meters")
    
    # Calculate distance using Simpson's Rule
    print("\n" + "-" * 60)
    print("Calculating distance using Simpson's Rule...")
    total_distance_simp, cumulative_distance_simp = simpsons_rule(time, speed)
    print(f"Total Distance (Simpson's): {total_distance_simp:.2f} meters")
    
    # Compare methods
    print("\n" + "-" * 60)
    print("Method Comparison:")
    print(f"Trapezoidal Rule: {total_distance_trap:.2f} m")
    print(f"Simpson's Rule:   {total_distance_simp:.2f} m")
    print(f"Absolute Difference: {abs(total_distance_trap - total_distance_simp):.2f} m")
    print(f"Relative Difference: {abs(total_distance_trap - total_distance_simp) / total_distance_trap * 100:.2f}%")
    
    # Visualize results
    print("\n" + "-" * 60)
    print("Generating visualizations...")
    plot_results(time, speed, cumulative_distance_trap, cumulative_distance_simp,
                 total_distance_trap, total_distance_simp)
    print("Results saved to 'vehicle_distance_tracker_results.png'")
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
