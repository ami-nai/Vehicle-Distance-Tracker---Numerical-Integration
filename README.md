# Vehicle Distance Tracker

A Python application that calculates total distance traveled by a vehicle using numerical integration methods (Trapezoidal and Simpson's Rule) from discrete speed-time data.

## Features

- **Trapezoidal Rule Integration**: Approximates distance by treating speed curve as connected trapezoids
- **Simpson's 1/3 Rule Integration**: More accurate approximation using parabolic segments
- **Data Visualization**: Plots speed vs time and cumulative distance vs time
- **Method Comparison**: Compares accuracy between both numerical methods

## Project Structure

```
Vehicle Distance Tracker/
├── data/
│   └── speed_time_data.csv      # Sample speed-time data
├── vehicle_distance_tracker.py   # Main application
└── README.md                      # Project documentation
```

## Requirements

- Python 3.7+
- numpy
- pandas
- matplotlib

## Installation

```bash
pip install numpy pandas matplotlib
```

## Usage

Run the main script:

```bash
python vehicle_distance_tracker.py
```

The program will:
1. Load speed-time data from CSV
2. Calculate total distance using both methods
3. Display comparison results in terminal
4. Generate visualization plots
5. Save results as `vehicle_distance_tracker_results.png`

## Data Format

The CSV file should have two columns:
- `Time (s)`: Time in seconds
- `Speed (m/s)`: Speed in meters per second

Example:
```csv
Time (s),Speed (m/s)
0,0
1,2
2,4
3,5
```

## Numerical Methods

### Trapezoidal Rule
$$\text{Distance} = \sum_{i=1}^{n} \frac{h}{2}(v_i + v_{i-1})$$

### Simpson's 1/3 Rule
$$\text{Distance} = \sum_{i=2,4,6,...}^{n} \frac{h}{6}(v_{i-2} + 4v_{i-1} + v_i)$$

Where:
- $h$ = time interval
- $v$ = speed at each point
- $n$ = number of data points

## Output

The program generates:
- Terminal output with calculated distances and comparison
- PNG image with two subplots:
  1. Speed vs Time (with filled area)
  2. Cumulative Distance vs Time (comparing both methods)

## Applications

- GPS navigation distance calculation
- Fitness tracking and activity monitoring
- Automotive telemetry analysis
- Vehicle performance testing

## Author

IIUC - 7th Semester Numerical Methods Lab Project

## License

Educational Project
