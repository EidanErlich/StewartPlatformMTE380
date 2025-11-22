#!/usr/bin/env python3
"""
3D Stewart Platform PID Controller
Real-time 2D PID control for ball balancing on a 3-motor Stewart platform.

This controller integrates:
- Ball detection (BallDetector from ball_detection.py) for real-time vision
- Simple PID control for X and Y axes with live tuning
- Normal vector computation from PID outputs
- Arduino servo control (based on arduino_controller.py)
- Inverse kinematics (from inverseKinematics.py) for servo angle calculation
- Live plotting for PID tuning visualization (last 10 seconds)

Control Features:
- Rate-limited setpoints: Smooth reference changes to prevent control spikes
- Filtered normal vector: First-order smoothing on platform tilt commands
- Trigonometric normal mapping: Proper sin/cos geometry for accurate tilt control
- Auto-reset on parameter changes: Clears accumulated errors when tuning
- Real-time plotting: Visualize position, error, and control output for tuning

Features:
- Reads ball position (x, y) in meters from camera with calibration
- Uses independent PID controllers for x and y axes
- Maps PID outputs to a desired platform normal vector using real trigonometry
- Provides live tuning via text-based control panel
- Allows target selection via mouse clicks on camera feed
- Sends computed servo angles to Arduino hardware
- Non-blocking serial communication for minimal latency
- Live plots showing position, error, and PID output (10-second window)

Hardware Interface:
- Connects to Arduino via serial (auto-detects port)
- Uses StewartPlatform inverse kinematics to compute servo angles
- Falls back to simulation mode if Arduino not connected

Usage:
    python PID_3d.py                     # Run with live plotting
    python PID_3d.py --no-plot           # Run without plotting (lower CPU usage)
    python PID_3d.py cal/camera_calib.npz  # Run with camera calibration
    
Controls:
    - Click on camera window to set target position
    - W/S or UP/DOWN arrows to select parameter in control panel
    - ENTER to edit selected parameter, type value and ENTER to confirm
    - Press 'q' to quit, 'r' to manually reset PID integrals
    - PID automatically resets when Kp/Ki/Kd/TiltGain changed
    - Live plot updates automatically every 5 iterations
"""

import cv2
import numpy as np
import time
import argparse
import serial
import serial.tools.list_ports
from typing import Optional, Tuple
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import os
from datetime import datetime

# Import ball detection and inverse kinematics
from ball_detection import BallDetector
from inverseKinematics import StewartPlatform


class LivePlotter:
    """
    Real-time plotter for PID tuning visualization.
    Plots magnitude of position/error and 2D control vector for last 10 seconds.
    """
    
    def __init__(self, window_size: float = 10.0):
        """
        Initialize live plotter.
        
        Args:
            window_size: Time window to display in seconds (default: 10.0)
        """
        self.window_size = window_size
        
        # Data buffers (deques for efficient append/pop)
        self.times = deque(maxlen=1000)
        self.x_positions = deque(maxlen=1000)
        self.y_positions = deque(maxlen=1000)
        self.x_errors = deque(maxlen=1000)
        self.y_errors = deque(maxlen=1000)
        self.x_outputs = deque(maxlen=1000)
        self.y_outputs = deque(maxlen=1000)
        self.x_targets = deque(maxlen=1000)
        self.y_targets = deque(maxlen=1000)
        
        # Start time reference
        self.start_time = time.time()
        
        # Setup matplotlib for non-blocking interactive plotting
        plt.ion()
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title('PID Tuning Live Plot')
        
        # Create grid layout: 2 rows, 3 columns
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Row 0: Distance from target (magnitude)
        self.ax_distance = self.fig.add_subplot(gs[0, 0])
        self.ax_distance.set_title('Distance from Target', fontweight='bold')
        self.ax_distance.set_xlabel('Time (s)')
        self.ax_distance.set_ylabel('Distance (m)')
        self.ax_distance.grid(True, alpha=0.3)
        
        # Row 0: Error magnitude over time
        self.ax_error = self.fig.add_subplot(gs[0, 1])
        self.ax_error.set_title('Error Magnitude', fontweight='bold')
        self.ax_error.set_xlabel('Time (s)')
        self.ax_error.set_ylabel('Error (m)')
        self.ax_error.grid(True, alpha=0.3)
        
        # Row 0: Control effort magnitude
        self.ax_control_mag = self.fig.add_subplot(gs[0, 2])
        self.ax_control_mag.set_title('Control Effort', fontweight='bold')
        self.ax_control_mag.set_xlabel('Time (s)')
        self.ax_control_mag.set_ylabel('Output Magnitude')
        self.ax_control_mag.grid(True, alpha=0.3)
        
        # Row 1: 2D Position trajectory (bird's eye view)
        self.ax_trajectory = self.fig.add_subplot(gs[1, 0])
        self.ax_trajectory.set_title('Ball Trajectory (XY)', fontweight='bold')
        self.ax_trajectory.set_xlabel('X Position (m)')
        self.ax_trajectory.set_ylabel('Y Position (m)')
        self.ax_trajectory.grid(True, alpha=0.3)
        self.ax_trajectory.set_aspect('equal', adjustable='box')
        
        # Row 1: Control vector field (quiver plot showing recent control directions)
        self.ax_control_vector = self.fig.add_subplot(gs[1, 1])
        self.ax_control_vector.set_title('Control Vector History', fontweight='bold')
        self.ax_control_vector.set_xlabel('X Output')
        self.ax_control_vector.set_ylabel('Y Output')
        self.ax_control_vector.grid(True, alpha=0.3)
        self.ax_control_vector.axhline(0, color='k', linewidth=0.5)
        self.ax_control_vector.axvline(0, color='k', linewidth=0.5)
        self.ax_control_vector.set_aspect('equal', adjustable='box')
        
        # Row 1: Component breakdown (X vs Y outputs stacked)
        self.ax_components = self.fig.add_subplot(gs[1, 2])
        self.ax_components.set_title('Control Components', fontweight='bold')
        self.ax_components.set_xlabel('Time (s)')
        self.ax_components.set_ylabel('Output')
        self.ax_components.grid(True, alpha=0.3)
        
        # Initialize line objects
        self.lines = {}
        self.lines['distance'], = self.ax_distance.plot([], [], 'b-', linewidth=2, label='Distance')
        self.lines['error'], = self.ax_error.plot([], [], 'r-', linewidth=2, label='Error')
        self.lines['control_mag'], = self.ax_control_mag.plot([], [], 'm-', linewidth=2, label='Magnitude')
        
        # Trajectory lines
        self.lines['trajectory'], = self.ax_trajectory.plot([], [], 'b-', linewidth=1.5, alpha=0.6, label='Path')
        self.lines['current_pos'] = self.ax_trajectory.scatter([], [], c='blue', s=100, marker='o', label='Current', zorder=5)
        self.lines['target_pos'] = self.ax_trajectory.scatter([], [], c='green', s=150, marker='*', label='Target', zorder=5)
        self.ax_trajectory.legend(loc='upper right', fontsize=8)
        
        # Control vector scatter (color-coded by time)
        self.control_scatter = self.ax_control_vector.scatter([], [], c=[], cmap='viridis', s=30, alpha=0.6)
        
        # Component lines
        self.lines['x_output'], = self.ax_components.plot([], [], 'r-', linewidth=1.5, alpha=0.7, label='X Output')
        self.lines['y_output'], = self.ax_components.plot([], [], 'g-', linewidth=1.5, alpha=0.7, label='Y Output')
        self.ax_components.legend(loc='upper right', fontsize=8)
        
        plt.show(block=False)
        
    def add_data(self, x_pos: float, y_pos: float, x_target: float, y_target: float,
                 x_error: float, y_error: float, x_output: float, y_output: float):
        """
        Add new data point to the plot.
        
        Args:
            x_pos: Current x position (m)
            y_pos: Current y position (m)
            x_target: Target x position (m)
            y_target: Target y position (m)
            x_error: X-axis error (m)
            y_error: Y-axis error (m)
            x_output: X-axis PID output
            y_output: Y-axis PID output
        """
        current_time = time.time() - self.start_time
        
        # Add data to buffers
        self.times.append(current_time)
        self.x_positions.append(x_pos)
        self.y_positions.append(y_pos)
        self.x_targets.append(x_target)
        self.y_targets.append(y_target)
        self.x_errors.append(x_error)
        self.y_errors.append(y_error)
        self.x_outputs.append(x_output)
        self.y_outputs.append(y_output)
        
    def update_plot(self):
        """Update the plot with current data (non-blocking, optimized)."""
        if len(self.times) < 2:
            return
        
        # Convert deques to numpy arrays for plotting (only once)
        times_array = np.array(self.times)
        x_pos_array = np.array(self.x_positions)
        y_pos_array = np.array(self.y_positions)
        x_target_array = np.array(self.x_targets)
        y_target_array = np.array(self.y_targets)
        x_error_array = np.array(self.x_errors)
        y_error_array = np.array(self.y_errors)
        x_output_array = np.array(self.x_outputs)
        y_output_array = np.array(self.y_outputs)
        
        # Filter data to only show last window_size seconds
        current_time = times_array[-1]
        mask = times_array >= (current_time - self.window_size)
        
        times_windowed = times_array[mask]
        x_pos_windowed = x_pos_array[mask]
        y_pos_windowed = y_pos_array[mask]
        x_target_windowed = x_target_array[mask]
        y_target_windowed = y_target_array[mask]
        x_error_windowed = x_error_array[mask]
        y_error_windowed = y_error_array[mask]
        x_output_windowed = x_output_array[mask]
        y_output_windowed = y_output_array[mask]
        
        # Downsample for faster plotting (plot every 3rd point if > 300 points)
        if len(times_windowed) > 300:
            step = len(times_windowed) // 300
            times_windowed = times_windowed[::step]
            x_pos_windowed = x_pos_windowed[::step]
            y_pos_windowed = y_pos_windowed[::step]
            x_target_windowed = x_target_windowed[::step]
            y_target_windowed = y_target_windowed[::step]
            x_error_windowed = x_error_windowed[::step]
            y_error_windowed = y_error_windowed[::step]
            x_output_windowed = x_output_windowed[::step]
            y_output_windowed = y_output_windowed[::step]
        
        # Calculate magnitudes (vectorized)
        distance_from_target = np.sqrt(x_pos_windowed**2 + y_pos_windowed**2)
        error_magnitude = np.sqrt(x_error_windowed**2 + y_error_windowed**2)
        control_magnitude = np.sqrt(x_output_windowed**2 + y_output_windowed**2)
        
        # Update magnitude plots
        self.lines['distance'].set_data(times_windowed, distance_from_target)
        self.lines['error'].set_data(times_windowed, error_magnitude)
        self.lines['control_mag'].set_data(times_windowed, control_magnitude)
        
        # Update trajectory plot (2D bird's eye view)
        self.lines['trajectory'].set_data(x_pos_windowed, y_pos_windowed)
        if len(x_pos_windowed) > 0:
            self.lines['current_pos'].set_offsets([[x_pos_windowed[-1], y_pos_windowed[-1]]])
            self.lines['target_pos'].set_offsets([[x_target_windowed[-1], y_target_windowed[-1]]])
        
        # Update control vector scatter (show recent control directions, downsample)
        if len(x_output_windowed) > 0:
            # Only show last 100 points for scatter plot
            n_scatter = min(100, len(x_output_windowed))
            scatter_indices = np.linspace(0, len(x_output_windowed)-1, n_scatter, dtype=int)
            colors = times_windowed[scatter_indices] - times_windowed[0]
            self.control_scatter.set_offsets(np.c_[x_output_windowed[scatter_indices], 
                                                   y_output_windowed[scatter_indices]])
            self.control_scatter.set_array(colors)
        
        # Update component breakdown
        self.lines['x_output'].set_data(times_windowed, x_output_windowed)
        self.lines['y_output'].set_data(times_windowed, y_output_windowed)
        
        # Auto-scale axes with padding (only update limits, skip relim/autoscale for speed)
        self.ax_distance.set_xlim(current_time - self.window_size, current_time)
        if len(distance_from_target) > 0:
            y_min, y_max = np.min(distance_from_target), np.max(distance_from_target)
            y_pad = (y_max - y_min) * 0.1 or 0.01
            self.ax_distance.set_ylim(y_min - y_pad, y_max + y_pad)
        
        self.ax_error.set_xlim(current_time - self.window_size, current_time)
        if len(error_magnitude) > 0:
            y_min, y_max = np.min(error_magnitude), np.max(error_magnitude)
            y_pad = (y_max - y_min) * 0.1 or 0.01
            self.ax_error.set_ylim(y_min - y_pad, y_max + y_pad)
        
        self.ax_control_mag.set_xlim(current_time - self.window_size, current_time)
        if len(control_magnitude) > 0:
            y_min, y_max = np.min(control_magnitude), np.max(control_magnitude)
            y_pad = (y_max - y_min) * 0.1 or 0.01
            self.ax_control_mag.set_ylim(y_min - y_pad, y_max + y_pad)
        
        self.ax_components.set_xlim(current_time - self.window_size, current_time)
        if len(x_output_windowed) > 0:
            y_min = min(np.min(x_output_windowed), np.min(y_output_windowed))
            y_max = max(np.max(x_output_windowed), np.max(y_output_windowed))
            y_pad = (y_max - y_min) * 0.1 or 0.01
            self.ax_components.set_ylim(y_min - y_pad, y_max + y_pad)
        
        # Trajectory plot: auto-scale with some padding
        if len(x_pos_windowed) > 0:
            x_min = min(np.min(x_pos_windowed), np.min(x_target_windowed))
            x_max = max(np.max(x_pos_windowed), np.max(x_target_windowed))
            y_min = min(np.min(y_pos_windowed), np.min(y_target_windowed))
            y_max = max(np.max(y_pos_windowed), np.max(y_target_windowed))
            
            x_padding = (x_max - x_min) * 0.1 or 0.01
            y_padding = (y_max - y_min) * 0.1 or 0.01
            
            self.ax_trajectory.set_xlim(x_min - x_padding, x_max + x_padding)
            self.ax_trajectory.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Control vector plot: auto-scale with padding
        if len(x_output_windowed) > 0:
            output_range = max(np.max(np.abs(x_output_windowed)), np.max(np.abs(y_output_windowed)))
            if output_range > 0:
                self.ax_control_vector.set_xlim(-output_range * 1.1, output_range * 1.1)
                self.ax_control_vector.set_ylim(-output_range * 1.1, output_range * 1.1)
        
        # Redraw (use draw_idle for non-blocking, faster updates)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
    def close(self):
        """Close the plot window."""
        plt.close(self.fig)


class CSVLogger:
    """
    Efficient CSV logger for control loop data.
    Logs only measured values during runtime, calculates derived values on shutdown.
    """
    
    def __init__(self, runs_dir: str = "Runs"):
        """
        Initialize CSV logger.
        
        Args:
            runs_dir: Directory to save CSV files (default: "Runs")
        """
        self.runs_dir = runs_dir
        self.file_path = None
        self.csv_file = None
        self.writer = None
        self.start_time = None
        
        # Data buffer for measured values (to calculate derived values later)
        self.data_rows = []
        
        # Ensure Runs directory exists
        os.makedirs(runs_dir, exist_ok=True)
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(runs_dir, f"run_{timestamp}.csv")
        
        # Open file for writing
        self.csv_file = open(self.file_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        
        # Write header (measured values only, derived values will be added later)
        self.header = [
            'time', 'time_elapsed',
            'x_pos', 'y_pos',
            'x_target', 'y_target',
            'x_target_filtered', 'y_target_filtered',
            'x_error', 'y_error',
            'x_output', 'y_output',
            'normal_x', 'normal_y', 'normal_z',
            'Kp', 'Ki', 'Kd',
            'tilt_gain', 'max_tilt_angle', 'centered_tolerance',
            'ball_detected', 'ball_centered',
            'pid_x_integral', 'pid_y_integral',
            'pid_x_prev_error', 'pid_y_prev_error'
        ]
        self.writer.writerow(self.header)
        
        print(f"[CSV] Logging to: {self.file_path}")
    
    def log_measured(self, time_abs: float, time_elapsed: float,
                     x_pos: Optional[float], y_pos: Optional[float],
                     x_target: float, y_target: float,
                     x_target_filtered: float, y_target_filtered: float,
                     x_error: float, y_error: float,
                     x_output: float, y_output: float,
                     normal_x: float, normal_y: float, normal_z: float,
                     Kp: float, Ki: float, Kd: float,
                     tilt_gain: float, max_tilt_angle: float, centered_tolerance: float,
                     ball_detected: bool, ball_centered: bool,
                     pid_x_integral: float, pid_y_integral: float,
                     pid_x_prev_error: float, pid_y_prev_error: float):
        """
        Log measured values to CSV (efficient, no derived calculations).
        
        Args:
            time_abs: Absolute timestamp
            time_elapsed: Elapsed time since start
            x_pos, y_pos: Ball position (None if not detected)
            x_target, y_target: Target position
            x_target_filtered, y_target_filtered: Rate-limited target
            x_error, y_error: Position errors
            x_output, y_output: PID outputs
            normal_x, normal_y, normal_z: Normal vector components
            Kp, Ki, Kd: PID gains
            tilt_gain, max_tilt_angle, centered_tolerance: Control parameters
            ball_detected: Whether ball was detected
            ball_centered: Whether ball is centered
            pid_x_integral, pid_y_integral: PID integral states
            pid_x_prev_error, pid_y_prev_error: PID previous error states
        """
        # Store row data for later derived value calculation
        row_data = {
            'time': time_abs,
            'time_elapsed': time_elapsed,
            'x_pos': x_pos if x_pos is not None else '',
            'y_pos': y_pos if y_pos is not None else '',
            'x_target': x_target,
            'y_target': y_target,
            'x_target_filtered': x_target_filtered,
            'y_target_filtered': y_target_filtered,
            'x_error': x_error,
            'y_error': y_error,
            'x_output': x_output,
            'y_output': y_output,
            'normal_x': normal_x,
            'normal_y': normal_y,
            'normal_z': normal_z,
            'Kp': Kp,
            'Ki': Ki,
            'Kd': Kd,
            'tilt_gain': tilt_gain,
            'max_tilt_angle': max_tilt_angle,
            'centered_tolerance': centered_tolerance,
            'ball_detected': 1 if ball_detected else 0,
            'ball_centered': 1 if ball_centered else 0,
            'pid_x_integral': pid_x_integral,
            'pid_y_integral': pid_y_integral,
            'pid_x_prev_error': pid_x_prev_error,
            'pid_y_prev_error': pid_y_prev_error
        }
        self.data_rows.append(row_data)
        
        # Write measured values only (no derived calculations)
        row = [
            time_abs, time_elapsed,
            x_pos if x_pos is not None else '', y_pos if y_pos is not None else '',
            x_target, y_target,
            x_target_filtered, y_target_filtered,
            x_error, y_error,
            x_output, y_output,
            normal_x, normal_y, normal_z,
            Kp, Ki, Kd,
            tilt_gain, max_tilt_angle, centered_tolerance,
            1 if ball_detected else 0, 1 if ball_centered else 0,
            pid_x_integral, pid_y_integral,
            pid_x_prev_error, pid_y_prev_error
        ]
        self.writer.writerow(row)
    
    def finalize(self):
        """
        Calculate derived values and rewrite CSV with complete data.
        Called when control loop stops.
        """
        if not self.data_rows:
            self.close()
            return
        
        print("[CSV] Calculating derived values and finalizing log...")
        
        # Close the current file
        self.csv_file.close()
        
        # Reopen for writing (overwrite with complete data)
        self.csv_file = open(self.file_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        
        # Extended header with derived values
        extended_header = self.header + [
            'error_magnitude',
            'output_magnitude',
            'tilt_angle'
        ]
        self.writer.writerow(extended_header)
        
        # Calculate derived values and write complete rows
        for row_data in self.data_rows:
            # Calculate derived values
            x_error = row_data['x_error']
            y_error = row_data['y_error']
            x_output = row_data['x_output']
            y_output = row_data['y_output']
            normal_z = row_data['normal_z']
            
            # Error magnitude (always calculated from float values)
            error_magnitude = np.sqrt(float(x_error)**2 + float(y_error)**2)
            
            # Output magnitude (always calculated from float values)
            output_magnitude = np.sqrt(float(x_output)**2 + float(y_output)**2)
            
            # Tilt angle (degrees) from normal vector z-component
            tilt_angle = np.rad2deg(np.arccos(np.clip(float(normal_z), -1.0, 1.0)))
            
            # Write complete row
            row = [
                row_data['time'], row_data['time_elapsed'],
                row_data['x_pos'], row_data['y_pos'],
                row_data['x_target'], row_data['y_target'],
                row_data['x_target_filtered'], row_data['y_target_filtered'],
                row_data['x_error'], row_data['y_error'],
                row_data['x_output'], row_data['y_output'],
                row_data['normal_x'], row_data['normal_y'], row_data['normal_z'],
                row_data['Kp'], row_data['Ki'], row_data['Kd'],
                row_data['tilt_gain'], row_data['max_tilt_angle'], row_data['centered_tolerance'],
                row_data['ball_detected'], row_data['ball_centered'],
                row_data['pid_x_integral'], row_data['pid_y_integral'],
                row_data['pid_x_prev_error'], row_data['pid_y_prev_error'],
                error_magnitude,
                output_magnitude,
                tilt_angle
            ]
            self.writer.writerow(row)
        
        print(f"[CSV] Finalized log with {len(self.data_rows)} rows: {self.file_path}")
        self.close()
    
    def close(self):
        """Close CSV file."""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None


class PIDController:
    """
    Simple PID controller.
    """
    
    def __init__(self, Kp: float = 1.0, Ki: float = 0.0, Kd: float = 0.0,
                 output_limit: float = 1.0):
        """
        Initialize PID controller.
        
        Args:
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            output_limit: Maximum absolute value for PID output (saturation)
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limit = output_limit
        
        # Internal state
        self.integral = 0.0
        self.prev_error = 0.0
        
    def update(self, error: float, measurement: float, dt: float) -> float:
        """
        Update PID controller and return control output.
        
        Args:
            error: Current error (setpoint - measurement)
            measurement: Current measurement value (unused in simple version)
            dt: Time step in seconds
            
        Returns:
            Control output (clamped to output_limit)
        """
        if dt <= 0:
            dt = 0.001  # Prevent division by zero
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term
        self.integral += error * dt
        I = self.Ki * self.integral
        # Anti-windup: clamp the integral state so the I term cannot exceed the actuator limits
        if abs(self.Ki) > 1e-12:
            integral_limit = abs(self.output_limit) / abs(self.Ki)
            self.integral = np.clip(self.integral, -integral_limit, integral_limit)
            I = self.Ki * self.integral
        else:
            # Ki is zero (or effectively zero) — ensure no accidental large integral contribution
            self.integral = 0.0
            I = 0.0
        
        # Derivative term
        derivative = (error - self.prev_error) / dt
        D = self.Kd * derivative
        
        # Compute output
        output = P + I + D
        
        # Saturate output
        output = np.clip(output, -self.output_limit, self.output_limit)
        
        # Update state
        self.prev_error = error
        
        return output
    
    def reset(self):
        """Reset internal state."""
        self.integral = 0.0
        self.prev_error = 0.0
    
    def set_gains(self, Kp: float, Ki: float, Kd: float):
        """Update PID gains."""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd


class TwiddleOptimizer:
    """
    Coordinate Descent (Twiddle) optimizer for PID parameter tuning.
    Optimizes [Kp, Ki, Kd] by perturbing one parameter at a time.
    Note: TiltGain is excluded from optimization and should be set manually.
    """
    
    def __init__(self, initial_params: np.ndarray):
        """
        Initialize Twiddle optimizer.
        
        Args:
            initial_params: Initial parameter vector [Kp, Ki, Kd]
        """
        self.params = np.array(initial_params, dtype=float)
        
        # Perturbation vector (how much to change params by initially)
        # Adjusted based on analysis: Kd needs more exploration (optimal ~0.30-0.35)
        # Kd perturbation increased to encourage exploration of higher values
        self.d_params = np.array([0.05, 0.01, 0.08], dtype=float)  # Increased Kd from 0.05 to 0.08
        
        # Parameter bounds to prevent unsafe values
        self.param_bounds = np.array([
            [0.0, 50.0],   # Kp: [min, max]
            [0.0, 20.0],   # Ki: [min, max]
            [0.0, 10.0],   # Kd: [min, max]
        ])
        
        self.best_cost = float('inf')
        self.best_params = self.params.copy()  # Track best parameters found
        self.current_param_idx = 0
        self.stage = 0  # 0=try positive, 1=try negative
        self.iteration = 0
        
        # Exploration enhancement: track consecutive failures to detect local minima
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5  # After 5 consecutive failures, boost exploration
        self.param_failure_count = [0, 0, 0]  # Track failures per parameter
        self.iterations_since_improvement = 0  # Track global stagnation
        self.max_iterations_without_improvement = 20  # After 20 iterations, force exploration
        
    def get_params(self) -> np.ndarray:
        """Get current parameter vector [Kp, Ki, Kd]."""
        return self.params.copy()
    
    def report_cost(self, cost: float) -> np.ndarray:
        """
        Report cost from an episode and update parameters using Twiddle logic.
        
        Args:
            cost: Cost value from episode (lower is better)
            
        Returns:
            Next parameter vector to try
        """
        self.iteration += 1
        
        if cost < self.best_cost:
            # Improvement found - keep the change
            self.best_cost = cost
            # Store the parameters that achieved this best cost
            # Note: params are already updated with the perturbation, so they're the best
            self.best_params = self.params.copy()
            # Less aggressive growth to prevent overshooting (1.05 instead of 1.1)
            self.d_params[self.current_param_idx] *= 1.05
            
            # Reset failure tracking on success
            self.consecutive_failures = 0
            self.param_failure_count[self.current_param_idx] = 0
            self.iterations_since_improvement = 0
            
            # Move to next parameter
            self.current_param_idx = (self.current_param_idx + 1) % len(self.params)
            self.stage = 0
            
            print(f"[TWIDDLE] Iteration {self.iteration}: Cost improved to {cost:.5f}, "
                  f"moving to param {self.current_param_idx}")
        else:
            # No improvement
            self.consecutive_failures += 1
            self.param_failure_count[self.current_param_idx] += 1
            self.iterations_since_improvement += 1
            
            # Global stagnation detection: if no improvement for many iterations, boost all step sizes
            if self.iterations_since_improvement >= self.max_iterations_without_improvement:
                # Force exploration by increasing all step sizes
                for i in range(len(self.d_params)):
                    self.d_params[i] *= 1.3  # Boost all parameters
                print(f"[TWIDDLE] Global stagnation detected - boosting all step sizes for exploration")
                self.iterations_since_improvement = 0  # Reset counter
            
            if self.stage == 0:
                # Positive perturbation failed, try negative
                self.params[self.current_param_idx] -= 2 * self.d_params[self.current_param_idx]
                self.stage = 1
                print(f"[TWIDDLE] Iteration {self.iteration}: Positive failed, trying negative")
            else:
                # Both positive and negative failed, revert and shrink step size
                self.params[self.current_param_idx] += self.d_params[self.current_param_idx]
                
                # Less aggressive shrinkage to prevent getting stuck (0.95 instead of 0.9)
                # But if stuck for too long, boost exploration
                if self.param_failure_count[self.current_param_idx] >= self.max_consecutive_failures:
                    # Boost exploration: increase step size to escape local minimum
                    self.d_params[self.current_param_idx] *= 1.2
                    print(f"[TWIDDLE] Parameter {self.current_param_idx} stuck - boosting exploration "
                          f"(step size: {self.d_params[self.current_param_idx]:.5f})")
                    self.param_failure_count[self.current_param_idx] = 0  # Reset counter
                else:
                    self.d_params[self.current_param_idx] *= 0.95  # Gentle shrinkage
                
                # Special handling for Kd (index 2): encourage higher values
                # If Kd is low and failing, try a larger positive jump
                if self.current_param_idx == 2 and self.params[2] < 0.25:
                    # Kd is below optimal range, encourage exploration upward
                    if self.d_params[2] < 0.10:
                        self.d_params[2] = 0.10  # Ensure minimum step size for Kd
                        print(f"[TWIDDLE] Kd below optimal range - maintaining exploration step size")
                
                # Move to next parameter
                self.current_param_idx = (self.current_param_idx + 1) % len(self.params)
                self.stage = 0
                
                print(f"[TWIDDLE] Iteration {self.iteration}: Both directions failed, "
                      f"shrinking step size, moving to param {self.current_param_idx}")
        
        # Clamp parameters to bounds
        for i in range(len(self.params)):
            self.params[i] = np.clip(self.params[i], 
                                     self.param_bounds[i][0], 
                                     self.param_bounds[i][1])
        
        return self.prepare_next_attempt()
    
    def prepare_next_attempt(self) -> np.ndarray:
        """
        Apply perturbation for the next run.
        
        Returns:
            Parameter vector with perturbation applied
        """
        if self.stage == 0:
            perturbation = self.d_params[self.current_param_idx]
            
            # Special handling for Kd: bias toward higher values if below optimal range
            # Analysis shows optimal Kd is 0.30-0.35, so encourage exploration upward
            if self.current_param_idx == 2 and self.params[2] < 0.28:
                # Kd is below optimal range - use larger positive perturbation
                perturbation = max(perturbation, 0.10)  # Ensure at least 0.10 step
                if self.params[2] < 0.20:
                    perturbation *= 1.5  # Even larger step if very low
            
            self.params[self.current_param_idx] += perturbation
        
        # Clamp to bounds
        for i in range(len(self.params)):
            self.params[i] = np.clip(self.params[i], 
                                     self.param_bounds[i][0], 
                                     self.param_bounds[i][1])
        
        return self.params.copy()
    
    def get_status(self) -> dict:
        """Get current optimizer status for display."""
        param_names = ['Kp', 'Ki', 'Kd']  # TiltGain excluded from optimization
        return {
            'params': self.params.copy(),
            'best_params': self.best_params.copy(),  # Best parameters found so far
            'param_names': param_names,
            'current_param': param_names[self.current_param_idx],
            'best_cost': self.best_cost,
            'iteration': self.iteration,
            'd_params': self.d_params.copy()
        }


class AutoTuner:
    """
    Manages auto-tuning episodes and cost calculation.
    Integrates with ControlLoop to perform parameter optimization.
    """
    
    def __init__(self, state: 'ControlState', optimizer: TwiddleOptimizer, 
                 reset_callback=None):
        """
        Initialize auto-tuner.
        
        Args:
            state: ControlState instance
            optimizer: TwiddleOptimizer instance
            reset_callback: Callback function to reset PID controllers
        """
        self.state = state
        self.optimizer = optimizer
        self.reset_callback = reset_callback
        
        # Episode configuration
        self.episode_duration = 5.0  # seconds
        self.stabilization_duration = 2.0  # seconds to wait for ball to center before test
        
        # Episode state
        self.is_tuning = False
        self.episode_start_time = None
        self.stabilization_start_time = None
        self.episode_phase = "IDLE"  # IDLE, RECOVERING, STABILIZING, RUNNING, FINISHED
        
        # Active Recovery state
        self.recovery_start_time = None
        self.recovery_centered_start_time = None  # When ball first became centered
        self.recovery_hold_time = 2.0  # seconds - how long ball must stay centered
        self.recovery_timeout = 10.0  # seconds - max time to recover ball
        self.recovery_params = None  # Best-known PID params for recovery
        self.best_params_ever = None  # Store best parameters found so far
        self.best_cost_ever = float('inf')
        
        # Cost accumulation
        self.accumulated_error = 0.0
        self.accumulated_jitter = 0.0
        self.prev_ux = 0.0
        self.prev_uy = 0.0
        self.time_elapsed = 0.0
        
        # Safety limits
        self.max_error_safety = 0.15  # meters - abort if error exceeds this
        self.max_tilt_safety = 10.0  # degrees - abort if tilt exceeds this
        self.recovery_centered_threshold = 0.05  # meters - ball must be within 2cm to consider recovered
        
        # Cost function weights
        self.error_weight = 1.0  # Weight for ITAE term
        self.jitter_weight = 0.1  # Weight for control effort term
        
        # Test maneuver: hybrid approach - cycle through fixed test points
        # Covers all major directions for comprehensive testing
        # All points at 5cm radius for consistent evaluation
        self.test_points = [
            (0.05, 0.0),           # East
            (0.035355, -0.035355), # Southeast
            (0.0, -0.05),          # South
            (-0.035355, -0.035355),# Southwest
            (-0.05, 0.0),          # West
            (-0.035355, 0.035355), # Northwest
            (0.0, 0.05),           # North
            (0.035355, 0.035355),  # Northeast
        ]

        self.test_point_idx = 0  # Current test point index (0-7)
        
        # Parameter set evaluation tracking
        self.current_param_set = None  # Current parameter set being evaluated
        self.test_points_completed = 0  # How many test points completed for current param set
        self.accumulated_costs = []  # Costs from each test point for current param set
        self.param_set_start_time = None  # When current parameter set evaluation started
        
        # Episode statistics
        self.episode_count = 0
        self.last_cost = None
    
    def start_tuning(self):
        """Start the auto-tuning process."""
        if self.is_tuning:
            print("[AUTOTUNER] Already tuning!")
            return
        
        self.is_tuning = True
        self.episode_count = 0
        # Reset test point index to start from first test point
        self.test_point_idx = 0
        # Reset parameter set evaluation tracking
        self.current_param_set = None
        self.test_points_completed = 0
        self.accumulated_costs = []
        self.param_set_start_time = None
        # Initialize best params with current state values
        self.best_params_ever = np.array([self.state.Kp, self.state.Ki, self.state.Kd])
        self.best_cost_ever = float('inf')
        # Check if optimizer already has better params
        opt_status = self.optimizer.get_status()
        if opt_status['best_cost'] < float('inf'):
            # Optimizer has found some good params, use them
            self.best_params_ever = opt_status['best_params'].copy()
            self.best_cost_ever = opt_status['best_cost']
        self.recovery_params = self.best_params_ever.copy()
        print("[AUTOTUNER] Starting auto-tuning with comprehensive evaluation (all 8 test points per parameter set)...")
        # Start with recovery to ensure ball is centered
        self.start_recovery()
    
    def stop_tuning(self):
        """Stop the auto-tuning process."""
        if not self.is_tuning:
            return
        
        self.is_tuning = False
        self.episode_phase = "IDLE"
        print("[AUTOTUNER] Auto-tuning stopped")
    
    def start_episode(self):
        """Start a new tuning episode."""
        self.episode_count += 1
        
        # Reset target to center
        self.state.x_target = 0.0
        self.state.y_target = 0.0
        
        # Check if we need a new parameter set (after testing all 8 points)
        if self.test_points_completed >= len(self.test_points):
            # All test points completed for current parameter set
            # Report averaged cost to optimizer and get new parameters
            avg_cost = np.mean(self.accumulated_costs) if self.accumulated_costs else float('inf')
            print(f"[AUTOTUNER] Parameter set evaluation complete: "
                  f"Average cost across {len(self.accumulated_costs)} test points = {avg_cost:.5f}")
            
            # Report to optimizer and get next parameter set
            next_params = self.optimizer.report_cost(avg_cost)
            
            # Update best parameters if improved
            opt_status = self.optimizer.get_status()
            if opt_status['best_cost'] < self.best_cost_ever:
                self.best_cost_ever = opt_status['best_cost']
                self.best_params_ever = opt_status['best_params'].copy()
                self.recovery_params = self.best_params_ever.copy()
                print(f"[AUTOTUNER] New best cost: {self.best_cost_ever:.5f} with params: "
                      f"Kp={self.best_params_ever[0]:.5f}, Ki={self.best_params_ever[1]:.5f}, "
                      f"Kd={self.best_params_ever[2]:.5f}")
            
            # Reset for new parameter set
            new_params = next_params
            self.current_param_set = tuple(new_params)
            self.test_points_completed = 0
            self.accumulated_costs = []
            self.test_point_idx = 0  # Start from first test point
            self.param_set_start_time = time.time()
            
            print(f"[AUTOTUNER] Starting new parameter set evaluation: "
                  f"Kp={new_params[0]:.5f}, Ki={new_params[1]:.5f}, Kd={new_params[2]:.5f}")
        else:
            # Continue with same parameter set, just get current params
            new_params = self.optimizer.get_params()
            # Verify we're still using the same parameter set
            current_set = tuple(new_params)
            if self.current_param_set is None:
                self.current_param_set = current_set
                self.param_set_start_time = time.time()
            elif self.current_param_set != current_set:
                # Parameters changed unexpectedly - reset
                self.current_param_set = current_set
                self.test_points_completed = 0
                self.accumulated_costs = []
                self.test_point_idx = 0
        
        # Apply parameters
        self.state.Kp = new_params[0]
        self.state.Ki = new_params[1]
        self.state.Kd = new_params[2]
        # TiltGain is NOT updated - keep current manual value
        
        # Reset PID controllers
        if self.reset_callback is not None:
            self.reset_callback()
        
        # Start stabilization phase
        self.episode_phase = "STABILIZING"
        self.stabilization_start_time = time.time()
        self.episode_start_time = None
        
        # Reset cost accumulation for this test point
        self.accumulated_error = 0.0
        self.accumulated_jitter = 0.0
        self.prev_ux = 0.0
        self.prev_uy = 0.0
        self.time_elapsed = 0.0
        
        status = self.optimizer.get_status()
        print(f"[AUTOTUNER] Episode {self.episode_count}: Test point {self.test_points_completed + 1}/{len(self.test_points)} "
              f"for param set (Kp={new_params[0]:.5f}, Ki={new_params[1]:.5f}, Kd={new_params[2]:.5f})")
    
    def start_recovery(self):
        """Start active recovery phase using best-known PID parameters."""
        print(f"[AUTOTUNER] Starting ACTIVE RECOVERY with best params: "
              f"Kp={self.recovery_params[0]:.5f}, Ki={self.recovery_params[1]:.5f}, "
              f"Kd={self.recovery_params[2]:.5f}")
        
        # Set target to center
        self.state.x_target = 0.0
        self.state.y_target = 0.0
        
        # Use best-known parameters for recovery
        self.state.Kp = self.recovery_params[0]
        self.state.Ki = self.recovery_params[1]
        self.state.Kd = self.recovery_params[2]
        
        # Reset PID controllers
        if self.reset_callback is not None:
            self.reset_callback()
        
        # Enter recovery phase
        self.episode_phase = "RECOVERING"
        self.recovery_start_time = time.time()
        self.recovery_centered_start_time = None  # Reset centered tracking
    
    def update(self, error_x: float, error_y: float, 
               control_x: float, control_y: float, 
               tilt_angle: float, dt: float):
        """
        Update tuner with current control loop state.
        
        Args:
            error_x: X-axis error (m)
            error_y: Y-axis error (m)
            control_x: X-axis control output
            control_y: Y-axis control output
            tilt_angle: Current platform tilt angle (degrees)
            dt: Time step (s)
        """
        if not self.is_tuning:
            return
        
        current_time = time.time()
        error_magnitude = np.sqrt(error_x**2 + error_y**2)
        control_magnitude = np.sqrt(control_x**2 + control_y**2)
        
        # Safety check - abort episode if unsafe (only during RUNNING phase)
        if self.episode_phase == "RUNNING":
            if error_magnitude > self.max_error_safety or tilt_angle > self.max_tilt_safety:
                print(f"[AUTOTUNER] SAFETY ABORT: error={error_magnitude:.5f}m, "
                      f"tilt={tilt_angle:.5f}deg")
                self.finish_episode(float('inf'))
                return
        
        # Active Recovery Phase
        if self.episode_phase == "RECOVERING":
            # Check if ball is detected
            if self.state.ball_position is None:
                # Ball not detected - check timeout
                recovery_elapsed = current_time - self.recovery_start_time
                if recovery_elapsed > self.recovery_timeout:
                    print(f"[AUTOTUNER] RECOVERY FAILED: Ball not detected after {self.recovery_timeout:.1f}s")
                    print("[AUTOTUNER] ABORTING TUNING - Human intervention required")
                    self.stop_tuning()
                    return
                # Continue trying to recover
                return
            
            # Ball is detected - check if it's centered
            recovery_elapsed = current_time - self.recovery_start_time
            
            # Check if ball is within recovery threshold
            if error_magnitude < self.recovery_centered_threshold:
                # Ball is currently centered
                if self.recovery_centered_start_time is None:
                    # Ball just became centered - start tracking
                    self.recovery_centered_start_time = current_time
                
                # Check how long ball has been continuously centered
                centered_duration = current_time - self.recovery_centered_start_time
                
                # Ball must stay centered for the full hold time
                if centered_duration >= self.recovery_hold_time:
                    print(f"[AUTOTUNER] RECOVERY SUCCESS: Ball centered and stable for {centered_duration:.2f}s "
                          f"(total recovery time: {recovery_elapsed:.2f}s)")
                    self.start_episode()
                return
            else:
                # Ball is not centered - reset the centered start time
                if self.recovery_centered_start_time is not None:
                    self.recovery_centered_start_time = None
                
                # Still recovering - check timeout
                if recovery_elapsed > self.recovery_timeout:
                    print(f"[AUTOTUNER] RECOVERY FAILED: Ball not centered after {self.recovery_timeout:.1f}s")
                    print(f"[AUTOTUNER] Current error: {error_magnitude:.4f}m, threshold: {self.recovery_centered_threshold:.4f}m")
                    print("[AUTOTUNER] ABORTING TUNING - Human intervention required")
                    self.stop_tuning()
                    return
                # Continue recovery
                return
        
        if self.episode_phase == "STABILIZING":
            # Wait for ball to stabilize at center
            if current_time - self.stabilization_start_time < self.stabilization_duration:
                # Check if ball is centered (within tolerance)
                if error_magnitude < self.state.centered_tolerance:
                    # Ball is centered, can proceed
                    pass
                else:
                    # Still waiting for stabilization
                    return
            else:
                # Stabilization timeout - proceed anyway
                pass
            
            # Start the test maneuver - use current test point
            self.episode_phase = "RUNNING"
            self.episode_start_time = current_time
            # Use current test point (will increment after completion)
            current_test_idx = self.test_point_idx
            test_x, test_y = self.test_points[current_test_idx]
            self.state.x_target = test_x
            self.state.y_target = test_y
            print(f"[AUTOTUNER] Starting test maneuver: step to ({test_x:.5f}, {test_y:.5f})m "
                  f"(test point {current_test_idx + 1}/{len(self.test_points)})")
            return
        
        elif self.episode_phase == "RUNNING":
            # Accumulate cost during episode
            if self.episode_start_time is None:
                self.episode_start_time = current_time
            
            self.time_elapsed = current_time - self.episode_start_time
            
            # ITAE: Integral Time-weighted Absolute Error
            # Penalize errors that persist over time more heavily
            self.accumulated_error += (self.time_elapsed * error_magnitude) * dt
            
            # Control effort (jitter): penalize rapid control changes
            jitter_x = abs(control_x - self.prev_ux)
            jitter_y = abs(control_y - self.prev_uy)
            jitter_magnitude = np.sqrt(jitter_x**2 + jitter_y**2)
            self.accumulated_jitter += jitter_magnitude * dt
            
            self.prev_ux = control_x
            self.prev_uy = control_y
            
            # Check if episode duration reached
            if self.time_elapsed >= self.episode_duration:
                # Calculate total cost
                total_cost = (self.error_weight * self.accumulated_error + 
                             self.jitter_weight * self.accumulated_jitter)
                self.finish_episode(total_cost)
        
        elif self.episode_phase == "FINISHED":
            # Episode finished, waiting for next episode to start
            pass
    
    def finish_episode(self, cost: float):
        """
        Finish current episode (one test point) and prepare for next.
        
        Args:
            cost: Total cost for this test point
        """
        self.last_cost = cost
        self.episode_phase = "FINISHED"
        
        # Accumulate cost for current parameter set evaluation
        self.accumulated_costs.append(cost)
        self.test_points_completed += 1
        
        if cost == float('inf'):
            print(f"[AUTOTUNER] Test point {self.test_points_completed}/{len(self.test_points)} FAILED (safety abort)")
        else:
            print(f"[AUTOTUNER] Test point {self.test_points_completed}/{len(self.test_points)} complete: "
                  f"Cost={cost:.5f} (error={self.accumulated_error:.5f}, "
                  f"jitter={self.accumulated_jitter:.5f})")
        
        # Move to next test point for next episode
        self.test_point_idx = (self.test_point_idx + 1) % len(self.test_points)
        
        # Check if all test points completed for current parameter set
        if self.test_points_completed >= len(self.test_points):
            # All test points done - will report averaged cost in start_episode()
            avg_cost = np.mean(self.accumulated_costs)
            print(f"[AUTOTUNER] All {len(self.test_points)} test points completed. "
                  f"Average cost: {avg_cost:.5f} (range: {np.min(self.accumulated_costs):.5f} - {np.max(self.accumulated_costs):.5f})")
        
        # CRITICAL: Enter recovery phase instead of immediately starting next episode
        # This prevents the "death spiral" by ensuring ball is centered before next test
        # start_episode() will handle reporting to optimizer if all test points are done
        print("[AUTOTUNER] Test point finished. Entering ACTIVE RECOVERY phase...")
        self.start_recovery()
    
    def get_status(self) -> dict:
        """Get current tuning status for display."""
        if not self.is_tuning:
            return {
                'is_tuning': False,
                'phase': 'IDLE',
                'episode': 0
            }
        
        opt_status = self.optimizer.get_status()
        
        # Calculate time elapsed based on current phase
        if self.episode_phase == "RECOVERING" and self.recovery_start_time:
            recovery_time = time.time() - self.recovery_start_time
            # Calculate how long ball has been continuously centered
            if self.recovery_centered_start_time:
                centered_duration = time.time() - self.recovery_centered_start_time
            else:
                centered_duration = 0.0
        else:
            recovery_time = 0.0
            centered_duration = 0.0
        
        return {
            'is_tuning': True,
            'phase': self.episode_phase,
            'episode': self.episode_count,
            'time_elapsed': self.time_elapsed if self.episode_start_time else 0.0,
            'recovery_time': recovery_time,
            'recovery_centered_duration': centered_duration,
            'recovery_hold_time': self.recovery_hold_time,
            'current_param': opt_status['current_param'],
            'best_cost': opt_status['best_cost'],
            'iteration': opt_status['iteration'],
            'last_cost': self.last_cost,
            'accumulated_error': self.accumulated_error,
            'accumulated_jitter': self.accumulated_jitter,
            'test_points_completed': self.test_points_completed,
            'total_test_points': len(self.test_points),
            'avg_cost_current_set': np.mean(self.accumulated_costs) if self.accumulated_costs else None
        }


class ControlState:
    """
    Manages control state: gains, target position, and runtime parameters.
    
    TUNING GUIDE FOR FASTER RESPONSE:
    1. Increase Kp for faster initial reaction (but too high causes oscillation)
    2. Increase Kd to dampen oscillations from higher Kp
    3. Increase tilt_gain for more aggressive platform movement
    4. Increase Ki carefully to eliminate steady-state error (but causes overshoot)
    5. Increase output_limit in ControlLoop if PID is saturating
    
    Current settings optimized for: Fast response with moderate stability
    """
    
    def __init__(self):
        # Target position (meters, platform coordinates)
        self.x_target = 0.0
        self.y_target = 0.0
        
        # Rate-limited target position (smooth setpoint changes)
        self.x_target_filtered = 0.0
        self.y_target_filtered = 0.0
        self.max_target_rate = 5.0  # m/s - maximum rate of setpoint change
        
        # PID gains (same for both x and y axes) --> DEFAULT VALUES
        # Tuned for faster response while maintaining stability
        # self.Kp = 0.35  # Proportional gain - increased for faster reaction
        # self.Ki = 0.175   # Integral gain - increased to eliminate steady-state error faster
        # self.Kd = 0.300   # Derivative gain - increased for better damping at higher speeds

        self.Kp = 0.2975
        self.Ki = 0.165
        self.Kd = 0.39
        
        # Control parameters
        self.max_tilt_angle = 5.0  # degrees (maximum platform tilt)
        self.tilt_gain = 1.0  # Scaling factor from PID output to tilt - increased for faster response
        self.normal_filter_alpha = 1.0  # Smoothing on normal vector (0=no smoothing, 1=instant)
        
        # Runtime flags
        self.running = False
        self.emergency_stop = False
        
        # Current state
        self.current_normal = np.array([0.0, 0.0, 1.0])
        self.filtered_normal = np.array([0.0, 0.0, 1.0])  # Smoothed normal vector
        self.ball_position = None  # (x, y) or None if not detected
        self.ball_is_centered = False  # True if ball is within centered tolerance
        self.last_update_time = None
        
        # Centered margin: if error magnitude is below this, ball is considered centered
        # and control output is set to zero (prevents micro-adjustments)
        self.centered_tolerance = 0.015  # meters (3mm margin)
        
        # Live plotting toggle
        self.live_plotting_enabled = False  # Default to disabled (can be enabled via UI)
        
        # Auto-tuning toggle
        self.auto_tuning_enabled = False  # Default to disabled (can be enabled via UI)
        self.auto_tuner_status = None  # Status dict from AutoTuner (updated by ControlLoop)


class NormalController:
    """
    Maps PID control outputs to a desired platform normal vector using trigonometric geometry.
    
    Enhanced Features:
    - Real trigonometric mapping: θx, θy → (sin(θx), sin(θy), cos(θx)·cos(θy))
    - First-order filtering on normal vector for smooth platform motion
    - Proper normalization and numerical stability
    
    Sign Convention:
    - Positive PID output ux means "tilt platform to move ball toward +x"
    - Tilt angles θx and θy are directly proportional to PID outputs
    - Normal vector always has positive nz (platform facing up)
    """
    
    def __init__(self, tilt_gain: float = 0.5, max_tilt_angle_deg: float = 15.0,
                 filter_alpha: float = 1.0):
        """
        Initialize normal controller.
        
        Args:
            tilt_gain: Scaling factor from PID output to tilt angle (radians per unit)
            max_tilt_angle_deg: Maximum platform tilt angle in degrees
            filter_alpha: First-order filter coefficient (0=frozen, 1=instant, 0.3-0.5=smooth)
        """
        self.tilt_gain = tilt_gain
        self.max_tilt_angle_rad = np.deg2rad(max_tilt_angle_deg)
        self.filter_alpha = filter_alpha
        
        # Filtered state
        self.filtered_normal = np.array([0.0, 0.0, 1.0])
        
    def compute_normal(self, ux: float, uy: float) -> np.ndarray:
        """
        Compute desired platform normal from PID outputs using trigonometric geometry.
        
        Args:
            ux: PID output for x-axis (control demand)
            uy: PID output for y-axis (control demand)
            
        Returns:
            Filtered unit normal vector [nx, ny, nz] where ||n|| = 1 and nz > 0
            
        Algorithm:
        1. Scale PID outputs to tilt angles: θx = tilt_gain * ux, θy = tilt_gain * uy
        2. Clamp angles to maximum tilt
        3. Convert to normal using spherical geometry:
           nx = sin(θx)
           ny = sin(θy)
           nz = cos(θx) * cos(θy)
        4. Normalize to unit length
        5. Apply first-order filtering for smooth motion
        """
        # Scale PID outputs to tilt angles (radians)
        theta_x = self.tilt_gain * ux
        theta_y = self.tilt_gain * uy
        
        # Clamp to maximum tilt angle (preserve direction)
        tilt_magnitude = np.sqrt(theta_x**2 + theta_y**2)
        if tilt_magnitude > self.max_tilt_angle_rad:
            scale = self.max_tilt_angle_rad / tilt_magnitude
            theta_x *= scale
            theta_y *= scale
        
        # Convert tilt angles to normal vector using spherical geometry
        # This is the correct trigonometric mapping for small-to-moderate tilts
        nx_raw = np.sin(theta_x)
        ny_raw = np.sin(theta_y)
        nz_raw = np.cos(theta_x) * np.cos(theta_y)
        
        # Construct raw normal vector
        n_raw = np.array([nx_raw, ny_raw, nz_raw])
        
        # Normalize to unit length (numerical stability)
        n_norm = np.linalg.norm(n_raw)
        if n_norm > 1e-8:
            n_unit = n_raw / n_norm
        else:
            n_unit = np.array([0.0, 0.0, 1.0])  # Fallback to flat
        
        # Ensure nz is positive (platform facing up)
        if n_unit[2] < 0:
            n_unit = -n_unit
        
        # Apply first-order filter for smooth motion
        self.filtered_normal = (self.filter_alpha * n_unit + 
                               (1 - self.filter_alpha) * self.filtered_normal)
        
        # Re-normalize after filtering (important for stability)
        filter_norm = np.linalg.norm(self.filtered_normal)
        if filter_norm > 1e-8:
            self.filtered_normal = self.filtered_normal / filter_norm
        
        return self.filtered_normal
    
    def get_raw_normal(self, ux: float, uy: float) -> np.ndarray:
        """Get unfiltered normal vector (for debugging)."""
        theta_x = self.tilt_gain * ux
        theta_y = self.tilt_gain * uy
        
        tilt_magnitude = np.sqrt(theta_x**2 + theta_y**2)
        if tilt_magnitude > self.max_tilt_angle_rad:
            scale = self.max_tilt_angle_rad / tilt_magnitude
            theta_x *= scale
            theta_y *= scale
        
        nx = np.sin(theta_x)
        ny = np.sin(theta_y)
        nz = np.cos(theta_x) * np.cos(theta_y)
        
        n = np.array([nx, ny, nz])
        n_norm = np.linalg.norm(n)
        if n_norm > 1e-8:
            n = n / n_norm
        
        if n[2] < 0:
            n = -n
            
        return n
    
    def reset(self):
        """Reset filtered state to flat platform."""
        self.filtered_normal = np.array([0.0, 0.0, 1.0])
    
    def set_tilt_gain(self, gain: float):
        """Update tilt gain."""
        self.tilt_gain = gain
    
    def set_max_tilt_angle(self, angle_deg: float):
        """Update maximum tilt angle."""
        self.max_tilt_angle_rad = np.deg2rad(angle_deg)
    
    def set_filter_alpha(self, alpha: float):
        """Update filter coefficient (0=frozen, 1=instant)."""
        self.filter_alpha = np.clip(alpha, 0.0, 1.0)


# ============================================================================
# Hardware Interface - Arduino + Servo Control
# ============================================================================

class ArduinoServoController:
    """
    Controls Stewart platform servos via Arduino serial interface.
    Based on arduino_controller.py implementation.
    """
    
    def __init__(self, port=None, baud=115200):
        """
        Initialize Arduino servo controller.
        
        Args:
            port: Serial port (auto-detected if None)
            baud: Baud rate (default: 115200)
        """
        self.platform = StewartPlatform()
        self.ser = None
        self.connected = False
        
        # Find and connect to Arduino
        if port is None:
            port = self.find_arduino_port()
        
        if port:
            self.connect(port, baud)
        else:
            print("[ARDUINO] No Arduino found - running in simulation mode")
            self.connected = False
    
    def find_arduino_port(self):
        """Auto-detect Arduino port."""
        ports = serial.tools.list_ports.comports()
        for p in ports:
            # Look for common Arduino USB identifiers
            if 'usbmodem' in p.device or 'usbserial' in p.device or 'Arduino' in str(p.description):
                print(f"[ARDUINO] Found Arduino on: {p.device}")
                return p.device
        return None
    
    def connect(self, port, baud):
        """Connect to Arduino via serial."""
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            
            # Read startup messages
            while self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"[ARDUINO] {line}")
            
            print(f"[ARDUINO] Connected to Arduino on {port} at {baud} baud")
            print("[ARDUINO] Initializing all servos to 0 degrees...")
            # Send all servos to 0 degrees directly
            self.send_angles([0, 0, 0])
            print("[ARDUINO] Servos initialized: [0°, 0°, 0°]")
            self.connected = True
            
        except serial.SerialException as e:
            print(f"[ARDUINO] Failed to connect to {port}: {e}")
            print("[ARDUINO] Running in simulation mode")
            self.connected = False
    
    def send_angles(self, angles_deg):
        """Send servo angles to Arduino in degrees."""
        if self.ser and self.ser.is_open:
            # Format: "angle0,angle1,angle2\n"
            command = f"{int(angles_deg[0])},{int(angles_deg[1])},{int(angles_deg[2])}\n"
            self.ser.write(command.encode())
            
            # Non-blocking check for response - don't wait if nothing available
            # This reduces latency from 10ms to nearly zero
            if self.ser.in_waiting:
                response = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if response.startswith('OK:'):
                    return True
                elif response.startswith('ERR:'):
                    print(f"[ARDUINO] Error: {response}")
                    return False
            return True  # Assume success if no response yet (non-blocking)
        return False
    
    def set_normal(self, nx: float, ny: float, nz: float):
        """
        Set platform normal vector and compute/send servo angles.
        
        Args:
            nx: x-component of unit normal vector
            ny: y-component of unit normal vector
            nz: z-component of unit normal vector (should be positive)
        """
        normal = np.array([nx, ny, nz], dtype=float)
        
        # Calculate servo command angles (deg) with zero offsets and clamp to 0..65
        angles_deg = self.platform.calculate_servo_angles(
            normal,
            degrees=True,
            apply_offsets=True,
            clamp_min=0.0,
            clamp_max=65.0,
        )
        
        if self.connected:
            # Send to Arduino
            self.send_angles(angles_deg)
        
        # Log for debugging
        # print(f"[SERVO] Normal: [{nx:.3f}, {ny:.3f}, {nz:.3f}] | "
        #       f"Angles: [{angles_deg[0]:.1f}°, {angles_deg[1]:.1f}°, {angles_deg[2]:.1f}°]")
        
        return angles_deg
    
    def close(self):
        """Close serial connection."""
        if self.ser and self.ser.is_open:
            # Reset to neutral before closing
            self.set_normal(0.0, 0.0, 1.0)
            time.sleep(0.1)
            self.ser.close()
            print("[ARDUINO] Serial connection closed")


# ============================================================================
# Camera and Ball Tracking Wrapper
# ============================================================================

class CameraManager:
    """
    Manages camera capture and ball detection.
    Wraps BallDetector for integration with control loop.
    """
    
    def __init__(self, camera_id=0, config_file="config.json", calib_file=None):
        """
        Initialize camera and ball detector.
        
        Args:
            camera_id: Camera device ID (default: 0)
            config_file: Path to config.json
            calib_file: Path to camera calibration file (optional, None to disable calibration)
        """
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print(f"[CAMERA] Failed to open camera {camera_id}")
            self.cap = None
        else:
            # Set buffer size to 1 to always get the latest frame (reduces latency)
            # This prevents frame accumulation which causes choppy video
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except:
                pass  # Some cameras don't support this property
            print(f"[CAMERA] Opened camera {camera_id}")
        
        # Initialize ball detector
        self.detector = BallDetector(config_file=config_file, calib_file=calib_file)
        
        # Current state
        self.current_frame = None
        self.ball_position = None  # (x, y) in meters or None
        self.ball_detected = False
    
    def update(self):
        """
        Capture frame and detect ball.
        
        Returns:
            True if frame captured successfully, False otherwise
        """
        if self.cap is None:
            return False
        
        # Read latest frame (buffer size set to 1 in __init__ to reduce latency)
        ret, frame = self.cap.read()
        if not ret:
            return False
        
        # Apply camera undistortion if calibration is available
        if self.detector.has_camera_calib:
            if self.detector.newK is None:
                # Compute optimal new camera matrix once
                h, w = frame.shape[:2]
                self.detector.newK, roi = cv2.getOptimalNewCameraMatrix(
                    self.detector.K, self.detector.dist, (w, h), 1, (w, h)
                )
            frame = cv2.undistort(frame, self.detector.K, self.detector.dist, None, self.detector.newK)
        
        self.current_frame = frame
        
        # Detect ball (pass undistort=False since we already undistorted)
        found, center, radius, (x_m, y_m) = self.detector.detect_ball(frame, undistort=False)
        
        if found:
            self.ball_position = (x_m, y_m)
            self.ball_detected = True
        else:
            self.ball_position = None
            self.ball_detected = False
        
        return True
    
    def get_ball_position(self) -> Optional[Tuple[float, float]]:
        return self.ball_position
    
    def get_camera_frame(self) -> Optional[np.ndarray]:
        return self.current_frame
    
    def close(self):
        """Release camera."""
        if self.cap is not None:
            self.cap.release()
            print("[CAMERA] Camera released")


class UIManager:
    """
    Manages OpenCV UI: camera display, PID tuning trackbars, mouse callbacks.
    """
    
    def __init__(self, state: ControlState, camera_manager: CameraManager,
                 plate_radius_m: float = 0.15, frame_width: int = 1920, frame_height: int = 1440):
        """
        Initialize UI manager.
        
        Args:
            state: ControlState instance
            camera_manager: CameraManager instance
            plate_radius_m: Plate radius in meters (for pixel-to-platform conversion)
            frame_width: Camera frame width in pixels
            frame_height: Camera frame height in pixels
        """
        self.state = state
        self.camera_manager = camera_manager
        
        # Get calibration from ball detector
        detector = camera_manager.detector
        
        # Use calibrated coordinate frame if available, otherwise fallback
        if detector.has_coordinate_frame and detector.origin_px is not None:
            self.plate_center_px = tuple(detector.origin_px.astype(int))
            self.origin_px = detector.origin_px
            self.x_axis = detector.x_axis
            self.y_axis = detector.y_axis
            self.has_calibration = True
            print(f"[UI] Using calibrated coordinate frame, origin at {self.plate_center_px}")
        else:
            self.plate_center_px = (frame_width // 2, frame_height // 2)
            self.origin_px = np.array([frame_width / 2, frame_height / 2])
            self.x_axis = np.array([1.0, 0.0])
            self.y_axis = np.array([0.0, -1.0])
            self.has_calibration = False
            print(f"[UI] No calibration found, using frame center at {self.plate_center_px}")
        
        # Use pixel_to_meter_ratio from detector if available
        if detector.pixel_to_meter_ratio is not None:
            self.pixel_to_meter_ratio = detector.pixel_to_meter_ratio
            print(f"[UI] Using calibrated pixel_to_meter_ratio: {self.pixel_to_meter_ratio:.6f} m/px")
        else:
            # Fallback: estimate from plate radius
            self.plate_radius_px = min(frame_width, frame_height) * 0.4
            self.pixel_to_meter_ratio = plate_radius_m / self.plate_radius_px
            print(f"[UI] Using estimated pixel_to_meter_ratio: {self.pixel_to_meter_ratio:.6f} m/px")
        
        self.plate_radius_m = plate_radius_m
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # OpenCV windows
        self.window_name = "Stewart Platform Control"
        self.control_window = "PID Tuning"
        
        # Note: Display frame management removed - direct display in control loop
        
    def pixel_to_platform(self, u: int, v: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to platform coordinates (meters).
        Uses the same calibrated coordinate frame as ball detection.
        
        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate
            
        Returns:
            (x, y) in meters, platform coordinates
        """
        if self.has_calibration:
            # Use calibrated coordinate frame (same as ball detector)
            ball_px = np.array([u, v], dtype=np.float64)
            delta_px = ball_px - self.origin_px
            
            # Project onto calibrated axes
            x_m = np.dot(delta_px, self.x_axis) * self.pixel_to_meter_ratio
            y_m = np.dot(delta_px, self.y_axis) * self.pixel_to_meter_ratio
        else:
            # Fallback: simple linear mapping from frame center
            dx_px = u - self.plate_center_px[0]
            dy_px = self.plate_center_px[1] - v  # Invert y (image y increases downward)
            
            x_m = dx_px * self.pixel_to_meter_ratio
            y_m = dy_px * self.pixel_to_meter_ratio
        
        return (x_m, y_m)
    
    def platform_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """
        Convert platform coordinates (meters) to pixel coordinates.
        Inverse of pixel_to_platform, used for drawing overlays.
        
        Args:
            x_m: X-coordinate in meters
            y_m: Y-coordinate in meters
            
        Returns:
            (u, v) pixel coordinates
        """
        if self.has_calibration:
            # Use calibrated coordinate frame
            # x_m = delta · x_axis * ratio  =>  delta · x_axis = x_m / ratio
            # y_m = delta · y_axis * ratio  =>  delta · y_axis = y_m / ratio
            # Solve for delta_px: delta = (x_m/ratio) * x_axis + (y_m/ratio) * y_axis
            delta_px = (x_m / self.pixel_to_meter_ratio) * self.x_axis + \
                      (y_m / self.pixel_to_meter_ratio) * self.y_axis
            pixel_pos = self.origin_px + delta_px
            u = int(pixel_pos[0])
            v = int(pixel_pos[1])
        else:
            # Fallback: simple linear mapping
            u = int(self.plate_center_px[0] + x_m / self.pixel_to_meter_ratio)
            v = int(self.plate_center_px[1] - y_m / self.pixel_to_meter_ratio)
        
        return (u, v)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to set target position."""
        if event == cv2.EVENT_LBUTTONDOWN:
            x_target, y_target = self.pixel_to_platform(x, y)
            self.state.x_target = x_target
            self.state.y_target = y_target
            print(f"[TARGET] New target set: ({x_target:.5f}, {y_target:.5f}) m")
    
    def create_control_panel(self):
        """Create control panel window for parameter adjustment."""
        cv2.namedWindow(self.control_window)
        self.selected_param = 0  # Index of currently selected parameter
        self.param_names = ['Kp', 'Ki', 'Kd', 'TiltGain', 'MaxTilt', 'CenterTol', 'LivePlot', 'AutoTune']
        self.editing_mode = False
        self.edit_buffer = ""
        self.reset_callback = None  # Callback to reset PIDs when parameters change
        self.plotting_callback = None  # Callback to enable/disable plotting
        self.tuning_callback = None  # Callback to start/stop auto-tuning
    
    def set_reset_callback(self, callback):
        """Set callback function to reset PID controllers when parameters change."""
        self.reset_callback = callback
    
    def set_plotting_callback(self, callback):
        """Set callback function to enable/disable plotting when LivePlot is toggled."""
        self.plotting_callback = callback
    
    def set_tuning_callback(self, callback):
        """Set callback function to start/stop auto-tuning when AutoTune is toggled."""
        self.tuning_callback = callback
        
    def get_param_value(self, param_name: str):
        """Get current value of a parameter. Returns float for numeric params, bool for LivePlot/AutoTune."""
        if param_name == 'Kp':
            return self.state.Kp
        elif param_name == 'Ki':
            return self.state.Ki
        elif param_name == 'Kd':
            return self.state.Kd
        elif param_name == 'TiltGain':
            return self.state.tilt_gain
        elif param_name == 'MaxTilt':
            return self.state.max_tilt_angle
        elif param_name == 'CenterTol':
            return self.state.centered_tolerance
        elif param_name == 'LivePlot':
            return self.state.live_plotting_enabled
        elif param_name == 'AutoTune':
            return self.state.auto_tuning_enabled
        return 0.0
    
    def is_boolean_param(self, param_name: str) -> bool:
        """Check if parameter is boolean (toggle) type."""
        return param_name == 'LivePlot' or param_name == 'AutoTune'
    
    def set_param_value(self, param_name: str, value):
        """Set value of a parameter and reset PID controllers to clear accumulated errors."""
        needs_reset = False  # Track if PID reset is needed
        
        if param_name == 'Kp':
            value = max(0.0, min(float(value), 50.0))
            self.state.Kp = value
            needs_reset = True
        elif param_name == 'Ki':
            value = max(0.0, min(float(value), 20.0))
            self.state.Ki = value
            needs_reset = True
        elif param_name == 'Kd':
            value = max(0.0, min(float(value), 10.0))
            self.state.Kd = value
            needs_reset = True
        elif param_name == 'TiltGain':
            value = max(0.0, min(float(value), 2.0))
            self.state.tilt_gain = value
            needs_reset = True
        elif param_name == 'MaxTilt':
            value = max(0.0, min(float(value), 30.0))
            self.state.max_tilt_angle = value
            # MaxTilt doesn't need PID reset
        elif param_name == 'CenterTol':
            value = max(0.0, min(float(value), 0.05))  # Max 5cm
            self.state.centered_tolerance = value
            # CenterTol doesn't need PID reset
        elif param_name == 'LivePlot':
            # Toggle boolean value
            self.state.live_plotting_enabled = bool(value)
            if self.plotting_callback is not None:
                self.plotting_callback(self.state.live_plotting_enabled)
            print(f"[PLOTTER] Live plotting {'ENABLED' if self.state.live_plotting_enabled else 'DISABLED'}")
            return  # No reset needed for plotting toggle
        elif param_name == 'AutoTune':
            # Toggle boolean value
            self.state.auto_tuning_enabled = bool(value)
            if self.tuning_callback is not None:
                self.tuning_callback(self.state.auto_tuning_enabled)
            print(f"[AUTOTUNER] Auto-tuning {'ENABLED' if self.state.auto_tuning_enabled else 'DISABLED'}")
            return  # No reset needed for tuning toggle
        
        # Reset PID controllers if a control parameter was changed
        if needs_reset and self.reset_callback is not None:
            self.reset_callback()
            print(f"[RESET] PID integrals cleared after parameter change")
    
    def update_control_panel(self):
        """Update and display the control panel."""
        # Create blank image for control panel (increased height for new parameter)
        panel = np.zeros((485, 500, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark gray background
        
        # Title
        cv2.putText(panel, "PID Control Panel", (150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Instructions
        y_pos = 60
        instructions = [
            "W/S or UP/DOWN: Select parameter",
            "ENTER: Edit value / Toggle checkbox",
            "Type number and press ENTER to set",
            "ESC: Cancel editing"
        ]
        for instruction in instructions:
            cv2.putText(panel, instruction, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += 20
        
        # Draw separator line
        cv2.line(panel, (10, y_pos), (490, y_pos), (100, 100, 100), 1)
        y_pos += 20
        
        # Display parameters
        for i, param_name in enumerate(self.param_names):
            is_selected = (i == self.selected_param)
            value = self.get_param_value(param_name)
            is_boolean = self.is_boolean_param(param_name)
            
            # Background highlight for selected parameter
            if is_selected:
                cv2.rectangle(panel, (5, y_pos - 20), (495, y_pos + 5), (80, 80, 80), -1)
            
            # Parameter name
            color = (0, 255, 255) if is_selected else (255, 255, 255)
            cv2.putText(panel, f"{param_name}:", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2 if is_selected else 1)
            
            # Parameter value display
            if is_boolean:
                # Boolean parameter - show checkbox
                checkbox_x = 250
                checkbox_y = y_pos - 12
                checkbox_size = 20
                
                # Draw checkbox border
                border_color = (0, 255, 0) if is_selected else (150, 150, 150)
                cv2.rectangle(panel, (checkbox_x, checkbox_y), 
                             (checkbox_x + checkbox_size, checkbox_y + checkbox_size), 
                             border_color, 2)
                
                # Draw checkmark if enabled
                if value:
                    # Draw filled rectangle with checkmark
                    cv2.rectangle(panel, (checkbox_x + 2, checkbox_y + 2), 
                                 (checkbox_x + checkbox_size - 2, checkbox_y + checkbox_size - 2), 
                                 (0, 255, 0), -1)
                    # Draw checkmark
                    cv2.line(panel, (checkbox_x + 5, checkbox_y + 10),
                            (checkbox_x + 9, checkbox_y + 15), (0, 0, 0), 2)
                    cv2.line(panel, (checkbox_x + 9, checkbox_y + 15),
                            (checkbox_x + 15, checkbox_y + 5), (0, 0, 0), 2)
                
                # Text label next to checkbox
                label_text = "ON" if value else "OFF"
                label_color = (0, 255, 0) if value else (100, 100, 100)
                if is_selected:
                    label_color = (0, 255, 255) if value else (255, 200, 0)
                cv2.putText(panel, label_text, (checkbox_x + checkbox_size + 10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2 if is_selected else 1)
            else:
                # Numeric parameter - show value
                if is_selected and self.editing_mode:
                    value_text = self.edit_buffer + "_"
                    value_color = (0, 255, 0)  # Green when editing
                else:
                    value_text = f"{value:.5f}"
                    value_color = (255, 255, 255)
                
                cv2.putText(panel, value_text, (250, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, value_color, 2 if is_selected else 1)
            
            y_pos += 35
        
        # Status message
        y_pos += 10
        selected_param_name = self.param_names[self.selected_param] if self.param_names else ""
        is_boolean_selected = self.is_boolean_param(selected_param_name)
        
        if self.editing_mode and not is_boolean_selected:
            status_msg = "EDITING MODE - Type value and press ENTER"
            status_color = (0, 255, 0)
        elif is_boolean_selected:
            status_msg = "Press ENTER to toggle checkbox"
            status_color = (0, 255, 255)
        else:
            status_msg = "Navigation Mode - Press ENTER to edit"
            status_color = (100, 100, 255)
        
        cv2.putText(panel, status_msg, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        cv2.imshow(self.control_window, panel)
    
    def handle_control_key(self, key: int) -> bool:
        """
        Handle keyboard input for control panel.
        
        Returns:
            True if key was handled, False otherwise
        """
        if self.editing_mode:
            # Editing mode - capture number input (only for numeric parameters)
            param_name = self.param_names[self.selected_param]
            is_boolean = self.is_boolean_param(param_name)
            
            if is_boolean:
                # Boolean parameters shouldn't be in editing mode, toggle instead
                self.editing_mode = False
                self.edit_buffer = ""
                current_value = self.get_param_value(param_name)
                self.set_param_value(param_name, not current_value)
                return True
            
            if key == 13:  # ENTER - confirm edit
                try:
                    value = float(self.edit_buffer)
                    self.set_param_value(param_name, value)
                    print(f"[PARAM] {param_name} set to {value:.5f}")
                except ValueError:
                    print(f"[ERROR] Invalid number: {self.edit_buffer}")
                self.editing_mode = False
                self.edit_buffer = ""
                return True
            elif key == 27:  # ESC - cancel edit
                self.editing_mode = False
                self.edit_buffer = ""
                print("[PARAM] Edit cancelled")
                return True
            elif key == 8 or key == 127:  # BACKSPACE or DELETE
                self.edit_buffer = self.edit_buffer[:-1]
                return True
            elif chr(key) in '0123456789.-':  # Valid number characters
                self.edit_buffer += chr(key)
                return True
        else:
            # Navigation mode
            # Handle arrow keys - use 'w' and 's' as alternatives
            if key == 82 or key == 0 or key == ord('w'):  # UP arrow or 'w'
                self.selected_param = (self.selected_param - 1) % len(self.param_names)
                return True
            elif key == 84 or key == 1 or key == ord('s'):  # DOWN arrow or 's'
                self.selected_param = (self.selected_param + 1) % len(self.param_names)
                return True
            elif key == 13:  # ENTER - start editing or toggle
                param_name = self.param_names[self.selected_param]
                is_boolean = self.is_boolean_param(param_name)
                
                if is_boolean:
                    # Toggle boolean parameter
                    current_value = self.get_param_value(param_name)
                    self.set_param_value(param_name, not current_value)
                    return True
                else:
                    # Start editing numeric parameter
                    current_value = self.get_param_value(param_name)
                    self.edit_buffer = f"{current_value:.5f}"
                    self.editing_mode = True
                    print(f"[PARAM] Editing {param_name} (current: {current_value:.5f})")
                    return True
        
        return False
    
    def draw_overlay(self, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Draw control overlay on camera frame.
        
        Args:
            frame: Input BGR frame (can be None)
            
        Returns:
            Frame with overlay drawn, or None if input is None
        """
        if frame is None:
            return None
        
        # Copy frame for overlay (required to avoid modifying original)
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        
        # Cache expensive calculations
        center_x, center_y = self.plate_center_px
        
        # Draw crosshair at calibrated platform center (not frame center)
        cv2.line(overlay, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 255), 2)
        cv2.line(overlay, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 255), 2)
        cv2.circle(overlay, (center_x, center_y), 5, (0, 255, 255), -1)
        cv2.putText(overlay, "Origin", (center_x + 10, center_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw coordinate axes if calibrated (simplified - no tick marks for performance)
        if self.has_calibration:
            axis_length = 80
            # X-axis (red)
            x_end = (int(center_x + self.x_axis[0] * axis_length),
                    int(center_y + self.x_axis[1] * axis_length))
            cv2.arrowedLine(overlay, (center_x, center_y), x_end, (0, 0, 255), 2, tipLength=0.3)
            cv2.putText(overlay, "X", (x_end[0] + 5, x_end[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Y-axis (green)
            y_end = (int(center_x + self.y_axis[0] * axis_length),
                    int(center_y + self.y_axis[1] * axis_length))
            cv2.arrowedLine(overlay, (center_x, center_y), y_end, (0, 255, 0), 2, tipLength=0.3)
            cv2.putText(overlay, "Y", (y_end[0] + 5, y_end[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Skip tick marks for performance - they're expensive to draw
        
        # Draw target position using calibrated conversion
        target_x_px, target_y_px = self.platform_to_pixel(self.state.x_target, self.state.y_target)
        if 0 <= target_x_px < w and 0 <= target_y_px < h:
            # Draw centered tolerance zone as a circle around target
            tolerance_radius_px = int(self.state.centered_tolerance / self.pixel_to_meter_ratio)
            cv2.circle(overlay, (target_x_px, target_y_px), tolerance_radius_px, (0, 200, 100), 1)
            
            cv2.drawMarker(overlay, (target_x_px, target_y_px), (0, 255, 0), 
                          cv2.MARKER_CROSS, 20, 2)
            cv2.putText(overlay, "Target", (target_x_px + 5, target_y_px - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw ball position if available (using calibrated conversion)
        if self.state.ball_position is not None:
            x, y = self.state.ball_position
            ball_x_px, ball_y_px = self.platform_to_pixel(x, y)
            if 0 <= ball_x_px < w and 0 <= ball_y_px < h:
                # Change ball color based on whether it's centered
                ball_color = (0, 255, 0) if self.state.ball_is_centered else (255, 0, 255)
                cv2.circle(overlay, (ball_x_px, ball_y_px), 10, ball_color, 2)
                status_text = "CENTERED" if self.state.ball_is_centered else f"({x:.5f}, {y:.5f})m"
                cv2.putText(overlay, f"Ball: {status_text}", 
                           (ball_x_px + 15, ball_y_px), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, ball_color, 2)
        
        # Draw commanded normal vector as arrow from origin
        n = self.state.current_normal
        if n is not None and np.linalg.norm(n) > 0:
            # Scale the normal vector for visualization (projection onto platform plane)
            # The normal vector points "up" from the platform, but we want to show tilt direction
            # Project the tilt onto the XY plane
            tilt_scale = 100  # pixels
            
            # Normal vector n = [nx, ny, nz]
            # Tilt direction in platform coordinates is -[nx, ny] (negative because normal tilts opposite to desired motion)
            tilt_x = n[0] * tilt_scale
            tilt_y = n[1] * tilt_scale
            
            # Convert tilt vector to pixel coordinates using calibrated axes
            if self.has_calibration:
                tilt_px = tilt_x * self.x_axis / self.pixel_to_meter_ratio + \
                         tilt_y * self.y_axis / self.pixel_to_meter_ratio
            else:
                tilt_px = np.array([tilt_x / self.pixel_to_meter_ratio, 
                                   -tilt_y / self.pixel_to_meter_ratio])
            
            normal_end = (int(center_x + tilt_px[0]), int(center_y + tilt_px[1]))
            
            # Draw arrow showing tilt direction (orange/yellow)
            cv2.arrowedLine(overlay, (center_x, center_y), normal_end, (0, 165, 255), 3, tipLength=0.3)
            cv2.putText(overlay, "Tilt", (normal_end[0] + 5, normal_end[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        # Draw status text
        y_offset = 30
        cv2.putText(overlay, f"Target: ({self.state.x_target:.5f}, {self.state.y_target:.5f}) m",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        
        if self.state.ball_position is not None:
            x, y = self.state.ball_position
            ex = self.state.x_target - x
            ey = self.state.y_target - y
            error_mag = np.sqrt(ex**2 + ey**2)
            error_color = (0, 255, 0) if self.state.ball_is_centered else (255, 255, 0)
            status_str = " [CENTERED]" if self.state.ball_is_centered else ""
            cv2.putText(overlay, f"Error: {error_mag:.5f}m ({ex:.5f}, {ey:.5f}){status_str}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, error_color, 2)
            y_offset += 25
        
        n = self.state.current_normal
        tilt_angle = np.rad2deg(np.arccos(np.clip(n[2], -1.0, 1.0)))
        cv2.putText(overlay, f"Normal: ({n[0]:.5f}, {n[1]:.5f}, {n[2]:.5f})",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        cv2.putText(overlay, f"Tilt: {tilt_angle:.5f} deg",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # PID gains
        y_offset += 30
        cv2.putText(
            overlay,
            f"Kp: {self.state.Kp:.5f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y_offset += 20
        cv2.putText(
            overlay,
            f"Ki: {self.state.Ki:.5f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y_offset += 20
        cv2.putText(
            overlay,
            f"Kd: {self.state.Kd:.5f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y_offset += 20
        cv2.putText(
            overlay,
            f"CenterTol: {self.state.centered_tolerance*1000:.5f}mm",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 200, 255),
            1,
        )
        
        # Auto-tuning status (if enabled)
        if self.state.auto_tuner_status is not None:
            tuner_status = self.state.auto_tuner_status
            if tuner_status['is_tuning']:
                y_offset += 30
                cv2.putText(
                    overlay,
                    f"AUTO-TUNING: Episode {tuner_status['episode']} ({tuner_status['phase']})",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
                y_offset += 20
                # Show parameter set evaluation progress
                test_progress = f"Test points: {tuner_status.get('test_points_completed', 0)}/{tuner_status.get('total_test_points', 8)}"
                if tuner_status.get('avg_cost_current_set') is not None:
                    test_progress += f" | Avg: {tuner_status['avg_cost_current_set']:.5f}"
                cv2.putText(
                    overlay,
                    test_progress,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )
                y_offset += 20
                cv2.putText(
                    overlay,
                    f"Param: {tuner_status['current_param']} | "
                    f"Best Cost: {tuner_status['best_cost']:.5f}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )
                if tuner_status['last_cost'] is not None:
                    y_offset += 20
                    cv2.putText(
                        overlay,
                        f"Last Cost: {tuner_status['last_cost']:.5f} | "
                        f"Time: {tuner_status['time_elapsed']:.5f}s",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )
                # Show recovery time if in recovery phase
                if tuner_status['phase'] == 'RECOVERING' and 'recovery_time' in tuner_status:
                    y_offset += 20
                    recovery_time = tuner_status['recovery_time']
                    centered_duration = tuner_status.get('recovery_centered_duration', 0.0)
                    hold_time = tuner_status.get('recovery_hold_time', 2.0)
                    
                    if centered_duration > 0:
                        # Ball is centered, show progress toward hold time
                        progress = min(centered_duration / hold_time, 1.0)
                        status_text = f"RECOVERY: Centered {centered_duration:.2f}s / {hold_time:.1f}s required"
                        # Color: green when close to completion, yellow otherwise
                        color = (0, 255, 0) if progress > 0.8 else (0, 255, 255)
                    else:
                        # Ball not centered yet
                        status_text = f"RECOVERY: {recovery_time:.2f}s (max {10.0:.1f}s) - Centering..."
                        color = (0, 255, 255)  # Yellow
                    
                    cv2.putText(
                        overlay,
                        status_text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

        # Instructions
        y_offset = h - 60
        cv2.putText(
            overlay,
            "Click to set target | 'q' quit | 'r' reset",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        return overlay


class ControlLoop:
    """
    Main control loop that coordinates all components.
    """
    
    def __init__(self, state: ControlState, camera_manager: CameraManager,
                 normal_controller: NormalController, ui_manager: UIManager,
                 servo_controller: ArduinoServoController, enable_plotting: bool = True):
        """
        Initialize control loop.
        
        Args:
            state: ControlState instance
            camera_manager: CameraManager instance
            normal_controller: NormalController instance
            ui_manager: UIManager instance
            servo_controller: ArduinoServoController instance
            enable_plotting: Enable live plotting for PID tuning (default: True)
        """
        self.state = state
        self.camera_manager = camera_manager
        self.normal_controller = normal_controller
        self.ui_manager = ui_manager
        self.servo_controller = servo_controller
        
        # PID controllers (same gains for both X and Y)
        self.pid_x = PIDController(
            Kp=state.Kp, Ki=state.Ki, Kd=state.Kd,
            output_limit=20.0
        )
        self.pid_y = PIDController(
            Kp=state.Kp, Ki=state.Ki, Kd=state.Kd,
            output_limit=20.0
        )
        
        # Live plotter for PID tuning
        self.enable_plotting = enable_plotting
        # Initialize plotter based on initial state (may be overridden by UI toggle)
        if enable_plotting and state.live_plotting_enabled:
            self.plotter = LivePlotter(window_size=10.0)
            print("[PLOTTER] Live plotting enabled")
        else:
            self.plotter = None
            if not enable_plotting:
                print("[PLOTTER] Live plotting disabled (command-line)")
            else:
                print("[PLOTTER] Live plotting disabled (UI default)")
        
        # CSV logger for data recording
        self.csv_logger = CSVLogger(runs_dir="Runs")
        self.csv_start_time = None
        
        # Auto-tuner (initialized with current parameters)
        # Only optimize Kp, Ki, Kd - TiltGain is set manually
        initial_params = np.array([state.Kp, state.Ki, state.Kd])
        self.optimizer = TwiddleOptimizer(initial_params)
        self.auto_tuner = AutoTuner(
            state=state,
            optimizer=self.optimizer,
            reset_callback=None  # Will be set in run()
        )
    
    def set_plotting_enabled(self, enabled: bool):
        """
        Enable or disable live plotting dynamically.
        
        Args:
            enabled: True to enable plotting, False to disable
        """
        if enabled and self.plotter is None:
            # Enable plotting - create plotter
            if self.enable_plotting:  # Only if command-line didn't disable it
                self.plotter = LivePlotter(window_size=10.0)
                print("[PLOTTER] Live plotting ENABLED")
        elif not enabled and self.plotter is not None:
            # Disable plotting - close and remove plotter
            self.plotter.close()
            self.plotter = None
            print("[PLOTTER] Live plotting DISABLED")
        
    def run(self):
        """Run main control loop."""
        print("="*70)
        print("3D Stewart Platform PID Controller")
        print("="*70)
        print("Controls:")
        print("  - Click on camera feed to set target position")
        print("  - Use UP/DOWN arrows to select parameter in control panel")
        print("  - Press ENTER to edit selected parameter")
        print("  - Type value and press ENTER to confirm")
        print("  - Press 'q' to quit, 'r' to reset PID integrals")
        print("  - PID integrals auto-reset when Kp/Ki/Kd/TiltGain changed")
        print("="*70)
        
        # Initialize UI
        cv2.namedWindow(self.ui_manager.window_name)
        cv2.setMouseCallback(self.ui_manager.window_name, self.ui_manager.mouse_callback)
        self.ui_manager.create_control_panel()
        
        # Set up callback for PID reset when parameters change
        def reset_pids():
            self.pid_x.reset()
            self.pid_y.reset()
            self.normal_controller.reset()
        self.ui_manager.set_reset_callback(reset_pids)
        
        # Set up callback for plotting toggle
        def toggle_plotting(enabled: bool):
            self.set_plotting_enabled(enabled)
        self.ui_manager.set_plotting_callback(toggle_plotting)
        
        # Set up callback for auto-tuning toggle
        def toggle_tuning(enabled: bool):
            if enabled:
                self.auto_tuner.start_tuning()
            else:
                self.auto_tuner.stop_tuning()
        self.ui_manager.set_tuning_callback(toggle_tuning)
        
        # Set reset callback for auto-tuner
        self.auto_tuner.reset_callback = reset_pids
        
        self.state.running = True
        self.state.last_update_time = time.time()
        self.csv_start_time = time.time()  # For elapsed time calculation
        
        # Initialize filtered setpoint to current target
        self.state.x_target_filtered = self.state.x_target
        self.state.y_target_filtered = self.state.y_target
        
        # Control loop counters for throttling updates
        display_counter = 0
        control_panel_counter = 0
        
        # Control loop
        while self.state.running and not self.state.emergency_stop:
            loop_start = time.time()
            
            # Update camera and ball detection (every iteration - critical for control)
            self.camera_manager.update()
            
            # Update control panel display (every 20 iterations to reduce overhead)
            control_panel_counter += 1
            if control_panel_counter % 20 == 0:  # Reduced from 10 to 20
                self.ui_manager.update_control_panel()
            
            # Update PID gains and normal controller parameters (only when changed)
            # Cache previous values to avoid unnecessary updates
            if not hasattr(self, '_last_Kp'):
                self._last_Kp = self.state.Kp
                self._last_Ki = self.state.Ki
                self._last_Kd = self.state.Kd
                self._last_tilt_gain = self.state.tilt_gain
                self._last_max_tilt = self.state.max_tilt_angle
                self._last_filter_alpha = self.state.normal_filter_alpha
            
            if (self._last_Kp != self.state.Kp or self._last_Ki != self.state.Ki or 
                self._last_Kd != self.state.Kd):
                self.pid_x.set_gains(self.state.Kp, self.state.Ki, self.state.Kd)
                self.pid_y.set_gains(self.state.Kp, self.state.Ki, self.state.Kd)
                self._last_Kp = self.state.Kp
                self._last_Ki = self.state.Ki
                self._last_Kd = self.state.Kd
            
            if self._last_tilt_gain != self.state.tilt_gain:
                self.normal_controller.set_tilt_gain(self.state.tilt_gain)
                self._last_tilt_gain = self.state.tilt_gain
            
            if self._last_max_tilt != self.state.max_tilt_angle:
                self.normal_controller.set_max_tilt_angle(self.state.max_tilt_angle)
                self._last_max_tilt = self.state.max_tilt_angle
            
            if self._last_filter_alpha != self.state.normal_filter_alpha:
                self.normal_controller.set_filter_alpha(self.state.normal_filter_alpha)
                self._last_filter_alpha = self.state.normal_filter_alpha
            
            # Get ball position
            ball_pos = self.camera_manager.get_ball_position()
            self.state.ball_position = ball_pos
            
            # Calculate dt from actual time elapsed
            current_time = time.time()
            dt = current_time - self.state.last_update_time
            self.state.last_update_time = current_time
            
            # Clamp dt to reasonable range (handle first iteration and long pauses)
            dt = np.clip(dt, 0.001, 0.1)
            
            # Apply rate limiting to target setpoint (smooth reference changes)
            max_delta = self.state.max_target_rate * dt
            
            # X-axis rate limiting
            dx_target = self.state.x_target - self.state.x_target_filtered
            if abs(dx_target) > max_delta:
                dx_target = np.sign(dx_target) * max_delta
            self.state.x_target_filtered += dx_target
            
            # Y-axis rate limiting
            dy_target = self.state.y_target - self.state.y_target_filtered
            if abs(dy_target) > max_delta:
                dy_target = np.sign(dy_target) * max_delta
            self.state.y_target_filtered += dy_target

            # Control update (only if ball detected)
            if ball_pos is not None:
                x, y = ball_pos

                # Compute errors using rate-limited setpoint
                ex = self.state.x_target_filtered - x
                ey = self.state.y_target_filtered - y
                
                # Check if ball is within centered tolerance
                error_magnitude = np.sqrt(ex**2 + ey**2)
                is_centered = error_magnitude < self.state.centered_tolerance
                self.state.ball_is_centered = is_centered
                
                # Update PID controllers
                # If ball is centered (within tolerance), output zero
                if is_centered:
                    # Ball is centered - set outputs to zero
                    ux = 0.0
                    uy = 0.0
                else:
                    # Ball is not centered - normal PID control
                    ux = self.pid_x.update(ex, x, dt)
                    uy = self.pid_y.update(ey, y, dt)
                
                # Add data to live plotter (only if enabled in state)
                if self.state.live_plotting_enabled and self.plotter is not None:
                    self.plotter.add_data(
                        x_pos=x,
                        y_pos=y,
                        x_target=self.state.x_target_filtered,
                        y_target=self.state.y_target_filtered,
                        x_error=ex,
                        y_error=ey,
                        x_output=ux,
                        y_output=uy
                    )
                
                # Compute desired normal (includes filtering)
                n = self.normal_controller.compute_normal(ux, uy)
                self.state.current_normal = n
                self.state.filtered_normal = self.normal_controller.filtered_normal
                
                # Update auto-tuner if enabled
                if self.state.auto_tuning_enabled:
                    tilt_angle = np.rad2deg(np.arccos(np.clip(n[2], -1.0, 1.0)))
                    self.auto_tuner.update(
                        error_x=ex,
                        error_y=ey,
                        control_x=ux,
                        control_y=uy,
                        tilt_angle=tilt_angle,
                        dt=dt
                    )
                    # Update status in state for UI display
                    self.state.auto_tuner_status = self.auto_tuner.get_status()
                else:
                    self.state.auto_tuner_status = None
                
                # Send to hardware via Arduino servo controller
                self.servo_controller.set_normal(n[0], n[1], n[2])
            else:
                # Ball not detected - set default values for logging
                x, y = None, None
                ex = 0.0  # No error calculation when ball not detected
                ey = 0.0
                ux = 0.0
                uy = 0.0
                self.state.ball_is_centered = False
            
            # Log measured values to CSV (every 5 iterations to reduce I/O overhead)
            # Critical for performance - disk I/O is slow
            if not hasattr(self, '_csv_counter'):
                self._csv_counter = 0
            self._csv_counter += 1
            if self._csv_counter % 5 == 0:  # Log every 5 iterations (~50-100Hz instead of 500Hz)
                time_elapsed = current_time - self.csv_start_time
                n = self.state.current_normal
                self.csv_logger.log_measured(
                    time_abs=current_time,
                    time_elapsed=time_elapsed,
                    x_pos=x,
                    y_pos=y,
                    x_target=self.state.x_target,
                    y_target=self.state.y_target,
                    x_target_filtered=self.state.x_target_filtered,
                    y_target_filtered=self.state.y_target_filtered,
                    x_error=ex,
                    y_error=ey,
                    x_output=ux,
                    y_output=uy,
                    normal_x=n[0],
                    normal_y=n[1],
                    normal_z=n[2],
                    Kp=self.state.Kp,
                    Ki=self.state.Ki,
                    Kd=self.state.Kd,
                    tilt_gain=self.state.tilt_gain,
                    max_tilt_angle=self.state.max_tilt_angle,
                    centered_tolerance=self.state.centered_tolerance,
                    ball_detected=(ball_pos is not None),
                    ball_centered=self.state.ball_is_centered,
                    pid_x_integral=self.pid_x.integral,
                    pid_y_integral=self.pid_y.integral,
                    pid_x_prev_error=self.pid_x.prev_error,
                    pid_y_prev_error=self.pid_y.prev_error
                )
            
            # Update live plot periodically (every 30 iterations to reduce CPU load)
            # Only update if plotting is enabled in state
            if self.state.live_plotting_enabled and self.plotter is not None:
                if not hasattr(self, '_plot_counter'):
                    self._plot_counter = 0
                self._plot_counter += 1
                if self._plot_counter % 30 == 0:  # Reduced from 5 to 30
                    self.plotter.update_plot()
            
            # Update display (every 5 iterations to reduce overhead - ~30-60fps is plenty for visualization)
            display_counter += 1
            if display_counter % 5 == 0:  # Reduced from 2 to 5 for better performance
                frame = self.camera_manager.get_camera_frame()
                if frame is not None:
                    overlay = self.ui_manager.draw_overlay(frame)
                    if overlay is not None:
                        cv2.imshow(self.ui_manager.window_name, overlay)
            
            # Handle keyboard input (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            
            # Try to handle key in control panel first
            if key != 255:  # 255 means no key pressed
                handled = self.ui_manager.handle_control_key(key)
                if not handled:
                    # Handle global keys if not handled by control panel
                    if key == ord('q'):
                        print("[STOP] Emergency stop requested")
                        self.state.emergency_stop = True
                        break
                    elif key == ord('r'):
                        print("[RESET] Resetting PID integrals")
                        self.pid_x.reset()
                        self.pid_y.reset()
        
        # Cleanup: set platform to flat
        print("[CLEANUP] Setting platform to flat")
        self.servo_controller.set_normal(0.0, 0.0, 1.0)
        
        # Finalize CSV log (calculate derived values)
        print("[CLEANUP] Finalizing CSV log...")
        self.csv_logger.finalize()
        
        self.camera_manager.close()
        if self.plotter is not None:
            self.plotter.close()
        cv2.destroyAllWindows()
        self.state.running = False

def main():
    """Main entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="3D Stewart Platform PID Controller with Ball Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python PID_3d.py                          # Run without camera calibration
  python PID_3d.py cal/camera_calib.npz      # Run with camera calibration
  python PID_3d.py --no-plot                # Run without live plotting
        """
    )
    parser.add_argument(
        'calib_file',
        nargs='?',
        default=None,
        help='Path to camera calibration file (.npz). If not provided, camera calibration is disabled.'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable live plotting (reduces CPU usage)'
    )
    args = parser.parse_args()
    
    print("="*70)
    print("3D Stewart Platform PID Controller with Ball Detection")
    print("="*70)
    
    if args.calib_file:
        print(f"[CONFIG] Camera calibration file: {args.calib_file}")
    else:
        print("[CONFIG] Camera calibration: DISABLED")
    
    if args.no_plot:
        print("[CONFIG] Live plotting: DISABLED")
    else:
        print("[CONFIG] Live plotting: ENABLED")
    print("="*70)
    
    # Initialize components
    state = ControlState()
    
    # Initialize hardware (Arduino servo controller)
    print("[INIT] Connecting to Arduino...")
    servo_controller = ArduinoServoController()
    
    # Initialize camera and ball detection
    print("[INIT] Initializing camera and ball detector...")
    camera_manager = CameraManager(camera_id=0, calib_file=args.calib_file)
    
    # Initialize normal controller
    normal_controller = NormalController(
        tilt_gain=state.tilt_gain,
        max_tilt_angle_deg=state.max_tilt_angle
    )
    
    # Initialize UI manager
    ui_manager = UIManager(
        state=state,
        camera_manager=camera_manager,
        plate_radius_m=0.15,  # Adjust based on your platform
        frame_width=1920,
        frame_height=1440
    )
    
    # Create and run control loop
    control_loop = ControlLoop(state, camera_manager, normal_controller, ui_manager, 
                               servo_controller, enable_plotting=not args.no_plot)
    
    try:
        control_loop.run()
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user")
        state.emergency_stop = True
        servo_controller.set_normal(0.0, 0.0, 1.0)
        # Finalize CSV log before cleanup
        if hasattr(control_loop, 'csv_logger'):
            control_loop.csv_logger.finalize()
        camera_manager.close()
        servo_controller.close()
    except Exception as e:
        print(f"[ERROR] Control loop failed: {e}")
        import traceback
        traceback.print_exc()
        servo_controller.set_normal(0.0, 0.0, 1.0)
        # Finalize CSV log before cleanup
        if hasattr(control_loop, 'csv_logger'):
            control_loop.csv_logger.finalize()
        camera_manager.close()
        servo_controller.close()


if __name__ == "__main__":
    main()
