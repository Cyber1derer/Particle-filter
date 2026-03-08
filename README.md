# Particle Filter for Robot Localization

This project is a implementation of a **particle filter** for mobile robot pose estimation.
Given odometry and range measurements to known landmarks, it estimates the robot state `(x, y, θ)` over time.

## What the project does

At each time step, the filter:
1. **Predicts** a new particle set from odometry (motion update).
2. **Scores** each particle using landmark range measurements (sensor update).
3. **Resamples** particles according to their likelihood.
4. **Visualizes** particles, landmarks, and the estimated pose.

The code is intended as a compact reference for learning the standard particle-filter loop.

## Project structure

- `particle_filter.py` – main particle-filter pipeline, plotting, motion/sensor/resampling steps.
- `read_data.py` – parsers for map and sensor log files.
- `data/world.dat` – landmark map.
- `data/sensor_data.dat` – odometry + sensor observations over time.

## How it works

- **State representation:** each particle is a dictionary with `x`, `y`, and `theta`.
- **Motion model:** odometry increments (`r1`, `t`, `r2`) are applied with Gaussian noise.
- **Sensor model:** particle likelihood is computed from the difference between measured and expected landmark ranges.
- **Resampling:** stochastic universal sampling keeps high-weight particles and removes low-weight ones.
- **Pose estimate:** mean position and circular mean orientation of all particles.

## Requirements

- Python 3
- `numpy`
- `scipy`
- `matplotlib`

Install dependencies:

```bash
pip install numpy scipy matplotlib
```

Run:

```bash
python particle_filter.py
```

A matplotlib window will show the evolving particle cloud and estimated heading.

## Notes

- The random seed is fixed in the script for reproducible runs.
- The dataset and map are local files under `data/` and are loaded by default.
