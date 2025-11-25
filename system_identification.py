#!/usr/bin/env python3
"""
System Identification Script for Stewart Platform Ball Balancing
Experimentally determines the plant transfer function: Platform Normal → Ball Position

This script automatically runs a comprehensive suite of tests including:
- Step responses at multiple amplitudes (5°, 8°, 12°) on both axes
- Sinusoidal tests at multiple frequencies (0.2, 0.5, 1.0, 1.5 Hz)
- Chirp tests (frequency sweep 0.1-2.5 Hz)
- Multi-sine tests (combined frequencies)
- PRBS tests (pseudo-random binary sequence)

The system plant maps:
- Input: Platform normal vector [nx, ny, nz] (tilt angles in radians)
- Output: Ball position [x, y] in meters

Usage:
    python system_identification.py                      # Run full test suite (~15-20 min)
    python system_identification.py --quick              # Run reduced test suite
    python system_identification.py --analyze data.csv   # Analyze existing data file
    
Output:
    - Multiple CSV files in logs/ directory (one per test)
    - Summary file with all results and statistics
    - Automatic analysis showing DC gain (K) and time constant (τ)
    - Linearity assessment across different amplitudes
"""

import cv2
import numpy as np
import time
import argparse
import serial
import serial.tools.list_ports
from typing import Optional, Tuple, List
import csv
import os
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit

# Import ball detection and inverse kinematics
from ball_detection import BallDetector
from inverseKinematics import StewartPlatform


class SystemIdentifier:
    """
    System identification for Stewart platform ball balancing.
    Applies test signals and records ball position response.
    """
    
    def __init__(self, calib_file=None, arduino_port=None, max_tilt_deg=15.0):
        """
        Initialize system identifier.
        
        Args:
            calib_file: Path to camera calibration file
            arduino_port: Serial port for Arduino (auto-detect if None)
            max_tilt_deg: Maximum tilt angle in degrees for safety
        """
        # Initialize ball detector
        self.detector = BallDetector(config_file="config.json", calib_file=calib_file)
        
        # Initialize platform kinematics
        self.platform = StewartPlatform()
        
        # Safety limits
        self.max_tilt_rad = np.radians(max_tilt_deg)
        
        # Arduino connection
        self.ser = None
        self.arduino_connected = False
        if arduino_port is None:
            arduino_port = self.find_arduino_port()
        if arduino_port:
            self.connect_arduino(arduino_port)
        else:
            print("WARNING: No Arduino found. Running in simulation mode.")
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        # Data logging
        self.data_log = []
        self.start_time = None
        
        # Current state
        self.current_normal = np.array([0.0, 0.0, 1.0])
        self.current_position = np.array([0.0, 0.0])
        self.current_velocity = np.array([0.0, 0.0])
        self.last_position_time = None
        
    def find_arduino_port(self):
        """Auto-detect Arduino port"""
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if 'usbmodem' in p.device or 'usbserial' in p.device or 'Arduino' in str(p.description):
                print(f"Found Arduino on: {p.device}")
                return p.device
        return None
    
    def connect_arduino(self, port, baud=115200):
        """Connect to Arduino via serial"""
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            time.sleep(2)  # Wait for Arduino to reset
            
            # Read startup messages
            while self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"Arduino: {line}")
            
            print(f"Connected to Arduino on {port}")
            self.arduino_connected = True
            
            # Initialize to zero
            self.send_normal(np.array([0.0, 0.0, 1.0]))
            time.sleep(0.5)
            
        except serial.SerialException as e:
            print(f"Failed to connect to {port}: {e}")
            self.arduino_connected = False
    
    def send_normal(self, normal):
        """Send platform normal to Arduino"""
        self.current_normal = normal / np.linalg.norm(normal)
        
        if self.arduino_connected and self.ser and self.ser.is_open:
            # Calculate servo angles
            angles_deg = self.platform.calculate_servo_angles(
                self.current_normal,
                degrees=True,
                apply_offsets=True,
                clamp_min=0.0,
                clamp_max=65.0,
            )
            
            # Send to Arduino
            command = f"{int(angles_deg[0])},{int(angles_deg[1])},{int(angles_deg[2])}\n"
            self.ser.write(command.encode())
    
    def get_ball_position(self):
        """
        Read camera frame and detect ball position.
        
        Returns:
            (found, position, velocity): Ball found flag, position [x, y] in meters, velocity [vx, vy]
        """
        ret, frame = self.cap.read()
        if not ret:
            return False, np.array([0.0, 0.0]), np.array([0.0, 0.0])
        
        found, center, radius, position_m = self.detector.detect_ball(frame)
        
        if found:
            current_time = time.time()
            
            # Calculate velocity
            if self.last_position_time is not None:
                dt = current_time - self.last_position_time
                if dt > 0:
                    velocity = (np.array(position_m) - self.current_position) / dt
                else:
                    velocity = self.current_velocity
            else:
                velocity = np.array([0.0, 0.0])
            
            self.current_position = np.array(position_m)
            self.current_velocity = velocity
            self.last_position_time = current_time
            
            return True, self.current_position, self.current_velocity
        
        return False, self.current_position, self.current_velocity
    
    def log_data(self, timestamp, normal, position, velocity):
        """Log data point"""
        # Extract tilt angles from normal
        nx, ny, nz = normal
        theta_x = np.arctan2(nx, nz)  # Tilt about x-axis
        theta_y = np.arctan2(ny, nz)  # Tilt about y-axis
        
        self.data_log.append({
            'time': timestamp,
            'normal_x': nx,
            'normal_y': ny,
            'normal_z': nz,
            'tilt_x_rad': theta_x,
            'tilt_y_rad': theta_y,
            'ball_x': position[0],
            'ball_y': position[1],
            'ball_vx': velocity[0],
            'ball_vy': velocity[1],
        })
    
    def save_data(self, filename=None):
        """Save logged data to CSV"""
        if filename is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sysid_{timestamp_str}.csv"
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        filepath = os.path.join("logs", filename)
        
        if not self.data_log:
            print("No data to save")
            return None
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.data_log[0].keys())
            writer.writeheader()
            writer.writerows(self.data_log)
        
        print(f"Data saved to {filepath}")
        return filepath
    
    def tilt_to_normal(self, theta_x, theta_y):
        """
        Convert tilt angles to normal vector.
        
        Args:
            theta_x: Tilt about x-axis (radians)
            theta_y: Tilt about y-axis (radians)
        
        Returns:
            normal: Unit normal vector [nx, ny, nz]
        """
        # Clamp to safety limits
        theta_x = np.clip(theta_x, -self.max_tilt_rad, self.max_tilt_rad)
        theta_y = np.clip(theta_y, -self.max_tilt_rad, self.max_tilt_rad)
        
        # Compute normal using trigonometry
        nx = np.sin(theta_x)
        ny = np.sin(theta_y)
        nz = np.sqrt(max(0, 1 - nx**2 - ny**2))
        
        return np.array([nx, ny, nz])
    
    def run_step_response(self, axis='x', amplitude_deg=10.0, duration=10.0, settling_time=2.0):
        """
        Run step response test on one axis.
        
        Args:
            axis: 'x' or 'y' axis to test
            amplitude_deg: Step amplitude in degrees
            duration: Test duration in seconds
            settling_time: Initial settling time before step
        
        Returns:
            filepath: Path to saved data file
        """
        print(f"\n=== Step Response Test: {axis.upper()} axis, {amplitude_deg}° ===")
        print(f"Duration: {duration}s, Settling time: {settling_time}s")
        
        self.data_log = []
        self.start_time = time.time()
        
        amplitude_rad = np.radians(amplitude_deg)
        
        # Initial settling period (zero tilt)
        print(f"Settling for {settling_time}s...")
        while time.time() - self.start_time < settling_time:
            normal = np.array([0.0, 0.0, 1.0])
            self.send_normal(normal)
            
            found, position, velocity = self.get_ball_position()
            if found:
                t = time.time() - self.start_time
                self.log_data(t, normal, position, velocity)
            
            time.sleep(0.05)
        
        # Apply step
        print(f"Applying step input of {amplitude_deg}°...")
        step_start = time.time()
        
        while time.time() - step_start < duration:
            if axis == 'x':
                normal = self.tilt_to_normal(amplitude_rad, 0.0)
            else:  # axis == 'y'
                normal = self.tilt_to_normal(0.0, amplitude_rad)
            
            self.send_normal(normal)
            
            found, position, velocity = self.get_ball_position()
            if found:
                t = time.time() - self.start_time
                self.log_data(t, normal, position, velocity)
            
            time.sleep(0.05)
        
        # Return to zero
        print("Returning to zero...")
        self.send_normal(np.array([0.0, 0.0, 1.0]))
        time.sleep(1.0)
        
        return self.save_data(f"step_response_{axis}_axis.csv")
    
    def run_sine_test(self, axis='x', amplitude_deg=8.0, frequency=0.5, duration=20.0):
        """
        Run sinusoidal test on one axis.
        
        Args:
            axis: 'x' or 'y' axis to test
            amplitude_deg: Sine amplitude in degrees
            frequency: Frequency in Hz
            duration: Test duration in seconds
        
        Returns:
            filepath: Path to saved data file
        """
        print(f"\n=== Sinusoidal Test: {axis.upper()} axis, {amplitude_deg}° @ {frequency} Hz ===")
        print(f"Duration: {duration}s")
        
        self.data_log = []
        self.start_time = time.time()
        
        amplitude_rad = np.radians(amplitude_deg)
        omega = 2 * np.pi * frequency
        
        while time.time() - self.start_time < duration:
            t = time.time() - self.start_time
            
            # Generate sinusoidal tilt
            tilt = amplitude_rad * np.sin(omega * t)
            
            if axis == 'x':
                normal = self.tilt_to_normal(tilt, 0.0)
            else:  # axis == 'y'
                normal = self.tilt_to_normal(0.0, tilt)
            
            self.send_normal(normal)
            
            found, position, velocity = self.get_ball_position()
            if found:
                self.log_data(t, normal, position, velocity)
            
            time.sleep(0.01)
        
        # Return to zero
        print("Returning to zero...")
        self.send_normal(np.array([0.0, 0.0, 1.0]))
        time.sleep(1.0)
        
        return self.save_data(f"sine_test_{axis}_axis_{frequency}Hz.csv")
    
    def run_chirp_test(self, axis='x', amplitude_deg=8.0, f0=0.1, f1=2.0, duration=30.0):
        """
        Run frequency sweep (chirp) test on one axis.
        
        Args:
            axis: 'x' or 'y' axis to test
            amplitude_deg: Chirp amplitude in degrees
            f0: Start frequency in Hz
            f1: End frequency in Hz
            duration: Test duration in seconds
        
        Returns:
            filepath: Path to saved data file
        """
        print(f"\n=== Chirp Test: {axis.upper()} axis, {amplitude_deg}° from {f0} to {f1} Hz ===")
        print(f"Duration: {duration}s")
        
        self.data_log = []
        self.start_time = time.time()
        
        amplitude_rad = np.radians(amplitude_deg)
        
        while time.time() - self.start_time < duration:
            t = time.time() - self.start_time
            
            # Generate chirp signal
            tilt = amplitude_rad * signal.chirp(t, f0, duration, f1, method='linear')
            
            if axis == 'x':
                normal = self.tilt_to_normal(tilt, 0.0)
            else:  # axis == 'y'
                normal = self.tilt_to_normal(0.0, tilt)
            
            self.send_normal(normal)
            
            found, position, velocity = self.get_ball_position()
            if found:
                self.log_data(t, normal, position, velocity)
            
            time.sleep(0.01)
        
        # Return to zero
        print("Returning to zero...")
        self.send_normal(np.array([0.0, 0.0, 1.0]))
        time.sleep(1.0)
        
        return self.save_data(f"chirp_test_{axis}_axis.csv")
    
    def run_multisine_test(self, axis='x', amplitude_deg=6.0, frequencies=[0.2, 0.5, 1.0], duration=30.0):
        """
        Run multi-frequency sinusoidal test on one axis.
        
        Args:
            axis: 'x' or 'y' axis to test
            amplitude_deg: Total amplitude in degrees
            frequencies: List of frequencies in Hz
            duration: Test duration in seconds
        
        Returns:
            filepath: Path to saved data file
        """
        print(f"\n=== Multi-Sine Test: {axis.upper()} axis, {amplitude_deg}° @ {frequencies} Hz ===")
        print(f"Duration: {duration}s")
        
        self.data_log = []
        self.start_time = time.time()
        
        amplitude_rad = np.radians(amplitude_deg) / len(frequencies)  # Split amplitude
        omegas = [2 * np.pi * f for f in frequencies]
        
        while time.time() - self.start_time < duration:
            t = time.time() - self.start_time
            
            # Generate multi-sine signal
            tilt = sum(amplitude_rad * np.sin(omega * t) for omega in omegas)
            
            if axis == 'x':
                normal = self.tilt_to_normal(tilt, 0.0)
            else:  # axis == 'y'
                normal = self.tilt_to_normal(0.0, tilt)
            
            self.send_normal(normal)
            
            found, position, velocity = self.get_ball_position()
            if found:
                self.log_data(t, normal, position, velocity)
            
            time.sleep(0.01)
        
        # Return to zero
        print("Returning to zero...")
        self.send_normal(np.array([0.0, 0.0, 1.0]))
        time.sleep(1.0)
        
        freqs_str = "_".join([str(f) for f in frequencies])
        return self.save_data(f"multisine_test_{axis}_axis_{freqs_str}Hz.csv")
    
    def run_prbs_test(self, axis='x', amplitude_deg=10.0, switching_time=0.5, duration=30.0):
        """
        Run pseudo-random binary sequence (PRBS) test on one axis.
        
        Args:
            axis: 'x' or 'y' axis to test
            amplitude_deg: PRBS amplitude in degrees
            switching_time: Time between switches in seconds
            duration: Test duration in seconds
        
        Returns:
            filepath: Path to saved data file
        """
        print(f"\n=== PRBS Test: {axis.upper()} axis, ±{amplitude_deg}° ===")
        print(f"Duration: {duration}s, Switching time: {switching_time}s")
        
        self.data_log = []
        self.start_time = time.time()
        
        amplitude_rad = np.radians(amplitude_deg)
        last_switch_time = 0
        current_level = 0
        
        while time.time() - self.start_time < duration:
            t = time.time() - self.start_time
            
            # Switch randomly at intervals
            if t - last_switch_time >= switching_time:
                current_level = np.random.choice([-1, 0, 1])
                last_switch_time = t
            
            tilt = current_level * amplitude_rad
            
            if axis == 'x':
                normal = self.tilt_to_normal(tilt, 0.0)
            else:  # axis == 'y'
                normal = self.tilt_to_normal(0.0, tilt)
            
            self.send_normal(normal)
            
            found, position, velocity = self.get_ball_position()
            if found:
                self.log_data(t, normal, position, velocity)
            
            time.sleep(0.05)
        
        # Return to zero
        print("Returning to zero...")
        self.send_normal(np.array([0.0, 0.0, 1.0]))
        time.sleep(1.0)
        
        return self.save_data(f"prbs_test_{axis}_axis.csv")
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        if self.ser and self.ser.is_open:
            # Return to zero before closing
            self.send_normal(np.array([0.0, 0.0, 1.0]))
            time.sleep(0.5)
            self.ser.close()
        cv2.destroyAllWindows()


def analyze_step_response(filepath):
    """
    Analyze step response data to extract system parameters.
    
    Fits first-order or second-order model to step response.
    """
    print(f"\n=== Analyzing Step Response: {filepath} ===")
    
    # Load data
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    time = data[:, 0]
    tilt_x = data[:, 4]
    tilt_y = data[:, 5]
    ball_x = data[:, 6]
    ball_y = data[:, 7]
    
    # Determine which axis was excited
    if np.std(tilt_x) > np.std(tilt_y):
        axis = 'x'
        tilt = tilt_x
        position = ball_x
    else:
        axis = 'y'
        tilt = tilt_y
        position = ball_y
    
    # Find step start (where tilt changes significantly)
    tilt_diff = np.abs(np.diff(tilt))
    step_idx = np.argmax(tilt_diff)
    
    if step_idx < 10:
        print("Could not identify step start")
        return
    
    # Extract step response portion
    t_step = time[step_idx:] - time[step_idx]
    u_step = tilt[step_idx:]
    y_step = position[step_idx:]
    
    # Remove initial value
    y_step = y_step - y_step[0]
    
    # Get step amplitude
    u_amplitude = np.mean(u_step[-100:]) if len(u_step) > 100 else np.mean(u_step)
    
    # Get steady-state output
    y_ss = np.mean(y_step[-100:]) if len(y_step) > 100 else np.mean(y_step)
    
    # Calculate DC gain
    K = y_ss / u_amplitude if u_amplitude != 0 else 0
    
    print(f"Axis: {axis}")
    print(f"Step amplitude: {np.degrees(u_amplitude):.2f}°")
    print(f"Steady-state position: {y_ss:.4f} m")
    print(f"DC Gain (K): {K:.4f} m/rad = {K * np.pi/180:.6f} m/deg")
    
    # Try to fit first-order model: y(t) = K * (1 - exp(-t/tau))
    try:
        def first_order(t, K_fit, tau):
            return K_fit * u_amplitude * (1 - np.exp(-t / tau))
        
        # Initial guess
        p0 = [K, 1.0]
        
        # Fit
        popt, pcov = curve_fit(first_order, t_step, y_step, p0=p0, maxfev=10000)
        K_fit, tau_fit = popt
        
        print(f"\nFirst-Order Model: G(s) = K / (τs + 1)")
        print(f"  K = {K_fit:.4f} m/rad")
        print(f"  τ = {tau_fit:.4f} s")
        print(f"  Settling time (2%): {4 * tau_fit:.4f} s")
        
        # Calculate fit quality (R²)
        y_fit = first_order(t_step, *popt)
        ss_res = np.sum((y_step - y_fit) ** 2)
        ss_tot = np.sum((y_step - np.mean(y_step)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        print(f"  R² = {r2:.4f}")
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(time, np.degrees(tilt), label='Tilt Input', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Tilt Angle (deg)')
        plt.title(f'Step Response - {axis.upper()} axis')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(t_step, y_step * 100, 'b-', label='Measured', linewidth=2)
        plt.plot(t_step, y_fit * 100, 'r--', label=f'First-Order Fit (τ={tau_fit:.2f}s)', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Ball Position (cm)')
        plt.title(f'Step Response (K={K_fit:.4f} m/rad, R²={r2:.3f})')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(filepath.replace('.csv', '_analysis.png'), dpi=150)
        print(f"Plot saved to {filepath.replace('.csv', '_analysis.png')}")
        plt.show()
        
    except Exception as e:
        print(f"Fitting error: {e}")


def analyze_frequency_response(filepath):
    """
    Analyze frequency response data (sine, chirp, or multisine).
    
    Computes frequency response (gain and phase) from input-output data.
    """
    print(f"\n=== Analyzing Frequency Response: {filepath} ===")
    
    # Load data
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    time = data[:, 0]
    tilt_x = data[:, 4]
    tilt_y = data[:, 5]
    ball_x = data[:, 6]
    ball_y = data[:, 7]
    
    # Determine which axis was excited
    if np.std(tilt_x) > np.std(tilt_y):
        axis = 'x'
        tilt = tilt_x
        position = ball_x
    else:
        axis = 'y'
        tilt = tilt_y
        position = ball_y
    
    # Compute sampling rate
    dt = np.median(np.diff(time))
    fs = 1.0 / dt
    
    print(f"Axis: {axis}")
    print(f"Sampling rate: {fs:.1f} Hz")
    print(f"Data points: {len(time)}")
    
    # Compute FFT
    n = len(time)
    freqs = np.fft.rfftfreq(n, dt)
    
    U_fft = np.fft.rfft(tilt)
    Y_fft = np.fft.rfft(position)
    
    # Compute frequency response
    H = Y_fft / (U_fft + 1e-10)
    
    # Magnitude and phase
    mag = np.abs(H)
    phase = np.angle(H, deg=True)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(time, np.degrees(tilt), label='Tilt Input', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Tilt Angle (deg)')
    plt.title(f'Frequency Response Test - {axis.upper()} axis')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(time, position * 100, label='Ball Position', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (cm)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.semilogx(freqs[1:], 20 * np.log10(mag[1:]), linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Bode Plot - Magnitude')
    plt.grid(True, which='both')
    plt.xlim([0.01, fs/2])
    
    plt.tight_layout()
    plt.savefig(filepath.replace('.csv', '_frequency.png'), dpi=150)
    print(f"Plot saved to {filepath.replace('.csv', '_frequency.png')}")
    plt.show()


def run_comprehensive_tests():
    """
    Run a comprehensive suite of system identification tests.
    Tests both axes with multiple amplitudes, frequencies, and test types.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE SYSTEM IDENTIFICATION SUITE")
    print("="*80)
    print("\nThis will run multiple tests to fully characterize the system plant.")
    print("Estimated total time: ~15-20 minutes\n")
    
    # Test configuration
    axes = ['x', 'y']
    step_amplitudes = [5.0, 8.0, 12.0]  # degrees
    sine_frequencies = [0.2, 0.5, 1.0, 1.5]  # Hz
    sine_amplitude = 8.0  # degrees
    
    all_results = []
    
    try:
        sysid = SystemIdentifier(calib_file='cal/camera_calib.npz', max_tilt_deg=15.0)
        
        # Allow system to settle initially
        print("\n[INIT] Allowing system to settle for 3 seconds...")
        time.sleep(3.0)
        
        # ===== STEP RESPONSE TESTS =====
        print("\n" + "="*80)
        print("PHASE 1: STEP RESPONSE TESTS (Linearity Check)")
        print("="*80)
        
        for axis in axes:
            for amplitude in step_amplitudes:
                print(f"\n--- Step Test: {axis.upper()} axis, {amplitude}° ---")
                
                try:
                    filepath = sysid.run_step_response(
                        axis=axis,
                        amplitude_deg=amplitude,
                        duration=12.0,
                        settling_time=2.0
                    )
                    
                    if filepath:
                        all_results.append({
                            'test': 'step',
                            'axis': axis,
                            'amplitude': amplitude,
                            'file': filepath
                        })
                        
                        # Brief pause between tests
                        print("Pausing 2s before next test...")
                        time.sleep(2.0)
                
                except Exception as e:
                    print(f"Error in step test: {e}")
                    continue
        
        # ===== SINUSOIDAL TESTS =====
        print("\n" + "="*80)
        print("PHASE 2: SINUSOIDAL TESTS (Frequency Response)")
        print("="*80)
        
        for axis in axes:
            for freq in sine_frequencies:
                print(f"\n--- Sine Test: {axis.upper()} axis, {freq} Hz ---")
                
                try:
                    filepath = sysid.run_sine_test(
                        axis=axis,
                        amplitude_deg=sine_amplitude,
                        frequency=freq,
                        duration=15.0
                    )
                    
                    if filepath:
                        all_results.append({
                            'test': 'sine',
                            'axis': axis,
                            'frequency': freq,
                            'file': filepath
                        })
                        
                        # Brief pause
                        print("Pausing 2s before next test...")
                        time.sleep(2.0)
                
                except Exception as e:
                    print(f"Error in sine test: {e}")
                    continue
        
        # ===== CHIRP TESTS =====
        print("\n" + "="*80)
        print("PHASE 3: CHIRP TESTS (Full Frequency Sweep)")
        print("="*80)
        
        for axis in axes:
            print(f"\n--- Chirp Test: {axis.upper()} axis, 0.1-2.5 Hz ---")
            
            try:
                filepath = sysid.run_chirp_test(
                    axis=axis,
                    amplitude_deg=8.0,
                    f0=0.1,
                    f1=2.5,
                    duration=30.0
                )
                
                if filepath:
                    all_results.append({
                        'test': 'chirp',
                        'axis': axis,
                        'file': filepath
                    })
                    
                    # Brief pause
                    print("Pausing 3s before next test...")
                    time.sleep(3.0)
            
            except Exception as e:
                print(f"Error in chirp test: {e}")
                continue
        
        # ===== MULTISINE TESTS =====
        print("\n" + "="*80)
        print("PHASE 4: MULTISINE TESTS (Combined Frequencies)")
        print("="*80)
        
        for axis in axes:
            print(f"\n--- MultiSine Test: {axis.upper()} axis ---")
            
            try:
                filepath = sysid.run_multisine_test(
                    axis=axis,
                    amplitude_deg=6.0,
                    frequencies=[0.2, 0.5, 1.0, 1.5],
                    duration=25.0
                )
                
                if filepath:
                    all_results.append({
                        'test': 'multisine',
                        'axis': axis,
                        'file': filepath
                    })
                    
                    # Brief pause
                    print("Pausing 2s before next test...")
                    time.sleep(2.0)
            
            except Exception as e:
                print(f"Error in multisine test: {e}")
                continue
        
        # ===== PRBS TESTS =====
        print("\n" + "="*80)
        print("PHASE 5: PRBS TESTS (Random Input)")
        print("="*80)
        
        for axis in axes:
            print(f"\n--- PRBS Test: {axis.upper()} axis ---")
            
            try:
                filepath = sysid.run_prbs_test(
                    axis=axis,
                    amplitude_deg=10.0,
                    switching_time=0.5,
                    duration=30.0
                )
                
                if filepath:
                    all_results.append({
                        'test': 'prbs',
                        'axis': axis,
                        'file': filepath
                    })
                    
                    # Brief pause
                    print("Pausing 2s before next test...")
                    time.sleep(2.0)
            
            except Exception as e:
                print(f"Error in PRBS test: {e}")
                continue
        
        sysid.cleanup()
        
        # ===== ANALYSIS PHASE =====
        print("\n" + "="*80)
        print("ANALYSIS PHASE: Processing All Results")
        print("="*80)
        
        step_results_summary = []
        
        for result in all_results:
            if result['test'] == 'step':
                print(f"\n>>> Analyzing: {result['file']}")
                try:
                    # Analyze step response
                    data = np.genfromtxt(result['file'], delimiter=',', skip_header=1)
                    time_data = data[:, 0]
                    tilt_x = data[:, 4]
                    tilt_y = data[:, 5]
                    ball_x = data[:, 6]
                    ball_y = data[:, 7]
                    
                    # Determine axis
                    if np.std(tilt_x) > np.std(tilt_y):
                        tilt = tilt_x
                        position = ball_x
                    else:
                        tilt = tilt_y
                        position = ball_y
                    
                    # Find step
                    tilt_diff = np.abs(np.diff(tilt))
                    step_idx = np.argmax(tilt_diff)
                    
                    if step_idx > 10:
                        t_step = time_data[step_idx:] - time_data[step_idx]
                        u_step = tilt[step_idx:]
                        y_step = position[step_idx:] - position[step_idx]
                        
                        u_amplitude = np.mean(u_step[-50:]) if len(u_step) > 50 else np.mean(u_step)
                        y_ss = np.mean(y_step[-50:]) if len(y_step) > 50 else np.mean(y_step)
                        K = y_ss / u_amplitude if u_amplitude != 0 else 0
                        
                        # Try to fit first-order model
                        try:
                            def first_order(t, K_fit, tau):
                                return K_fit * u_amplitude * (1 - np.exp(-t / tau))
                            
                            popt, _ = curve_fit(first_order, t_step, y_step, p0=[K, 1.0], maxfev=10000)
                            K_fit, tau_fit = popt
                            
                            step_results_summary.append({
                                'axis': result['axis'],
                                'amplitude_deg': result['amplitude'],
                                'K': K_fit,
                                'tau': tau_fit,
                                'file': result['file']
                            })
                            
                            print(f"  Axis: {result['axis']}, Amp: {result['amplitude']}° | K={K_fit:.4f} m/rad, τ={tau_fit:.3f}s")
                        
                        except Exception as e:
                            print(f"  Fit failed: {e}")
                
                except Exception as e:
                    print(f"  Analysis error: {e}")
        
        # ===== SUMMARY =====
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        
        print(f"\nTotal tests completed: {len(all_results)}")
        print(f"  - Step response: {len([r for r in all_results if r['test'] == 'step'])}")
        print(f"  - Sinusoidal: {len([r for r in all_results if r['test'] == 'sine'])}")
        print(f"  - Chirp: {len([r for r in all_results if r['test'] == 'chirp'])}")
        print(f"  - MultiSine: {len([r for r in all_results if r['test'] == 'multisine'])}")
        print(f"  - PRBS: {len([r for r in all_results if r['test'] == 'prbs'])}")
        
        if step_results_summary:
            print("\n--- Step Response Summary ---")
            for axis in axes:
                axis_results = [r for r in step_results_summary if r['axis'] == axis]
                if axis_results:
                    K_values = [r['K'] for r in axis_results]
                    tau_values = [r['tau'] for r in axis_results]
                    
                    print(f"\n{axis.upper()}-axis:")
                    print(f"  DC Gain (K):")
                    print(f"    Mean: {np.mean(K_values):.4f} m/rad")
                    print(f"    Std:  {np.std(K_values):.4f} m/rad")
                    print(f"    Range: [{np.min(K_values):.4f}, {np.max(K_values):.4f}]")
                    print(f"  Time Constant (τ):")
                    print(f"    Mean: {np.mean(tau_values):.3f} s")
                    print(f"    Std:  {np.std(tau_values):.3f} s")
                    print(f"    Range: [{np.min(tau_values):.3f}, {np.max(tau_values):.3f}]")
                    
                    # Linearity check
                    K_cv = np.std(K_values) / np.mean(K_values) * 100 if np.mean(K_values) != 0 else 0
                    print(f"  Linearity (K variation): {K_cv:.1f}% CoV")
                    if K_cv < 10:
                        print(f"    ✓ System appears LINEAR (K consistent across amplitudes)")
                    else:
                        print(f"    ⚠ System may be NONLINEAR (K varies with amplitude)")
        
        # Save summary to file
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"logs/sysid_summary_{timestamp_str}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("SYSTEM IDENTIFICATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total tests: {len(all_results)}\n\n")
            
            f.write("ALL TEST FILES:\n")
            for result in all_results:
                f.write(f"  - {result['file']}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("STEP RESPONSE ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            for axis in axes:
                axis_results = [r for r in step_results_summary if r['axis'] == axis]
                if axis_results:
                    f.write(f"\n{axis.upper()}-Axis:\n")
                    for r in axis_results:
                        f.write(f"  Amplitude: {r['amplitude_deg']}° | K={r['K']:.4f} m/rad | τ={r['tau']:.3f}s\n")
                    
                    K_values = [r['K'] for r in axis_results]
                    tau_values = [r['tau'] for r in axis_results]
                    f.write(f"\n  Average K: {np.mean(K_values):.4f} m/rad (±{np.std(K_values):.4f})\n")
                    f.write(f"  Average τ: {np.mean(tau_values):.3f} s (±{np.std(tau_values):.3f})\n")
        
        print(f"\nSummary saved to: {summary_file}")
        print("\n" + "="*80)
        print("ALL TESTS COMPLETE!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sysid.cleanup()
    except Exception as e:
        print(f"\nError during tests: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='System Identification for Stewart Platform')
    parser.add_argument('--analyze', type=str,
                        help='Analyze existing data file')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test suite (reduced tests for faster results)')
    
    args = parser.parse_args()
    
    # Analysis mode
    if args.analyze:
        if 'step' in args.analyze.lower():
            analyze_step_response(args.analyze)
        else:
            analyze_frequency_response(args.analyze)
        return
    
    # Run comprehensive test suite
    print("\nStarting comprehensive system identification...")
    print("Press Ctrl+C at any time to stop.\n")
    
    if args.quick:
        print("Quick mode: Running reduced test suite\n")
        # Could implement a quicker version here if needed
    
    run_comprehensive_tests()


if __name__ == "__main__":
    main()
