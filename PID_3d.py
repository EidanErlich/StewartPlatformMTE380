#!/usr/bin/env python3
"""
3D Stewart Platform PID Controller
Real-time 2D PID control for ball balancing on a 3-motor Stewart platform.

This controller integrates:
- Ball detection (BallDetector from ball_detection.py) for real-time vision
- Advanced PID control for X and Y axes with live tuning
- Normal vector computation from PID outputs
- Arduino servo control (based on arduino_controller.py)
- Inverse kinematics (from inverseKinematics.py) for servo angle calculation

Advanced Control Features:
- Back-calculation anti-windup: Prevents integral windup with proper feedback
- Derivative-on-error: Low-pass filtered d(error)/dt for better setpoint tracking
- Velocity estimation: Exponential moving average for smooth velocity feedback
- Rate-limited setpoints: Smooth reference changes to prevent control spikes
- Filtered normal vector: First-order smoothing on platform tilt commands
- Trigonometric normal mapping: Proper sin/cos geometry for accurate tilt control
- Auto-reset on parameter changes: Clears accumulated errors when tuning

Features:
- Reads ball position (x, y) in meters from camera with calibration
- Uses independent PID controllers for x and y axes
- Maps PID outputs to a desired platform normal vector using real trigonometry
- Provides live tuning via text-based control panel
- Allows target selection via mouse clicks on camera feed
- Sends computed servo angles to Arduino hardware
- Non-blocking serial communication for minimal latency

Hardware Interface:
- Connects to Arduino via serial (auto-detects port)
- Uses StewartPlatform inverse kinematics to compute servo angles
- Falls back to simulation mode if Arduino not connected

Usage:
    python PID_3d.py
    
Controls:
    - Click on camera window to set target position
    - W/S or UP/DOWN arrows to select parameter in control panel
    - ENTER to edit selected parameter, type value and ENTER to confirm
    - Press 'q' to quit, 'r' to manually reset PID integrals
    - PID automatically resets when Kp/Ki/Kd/TiltGain changed
"""

import cv2
import numpy as np
import time
import sys
import argparse
import serial
import serial.tools.list_ports
from typing import Optional, Tuple
from collections import deque

# Import ball detection and inverse kinematics
from ball_detection import BallDetector
from inverseKinematics import StewartPlatform


class PIDController:
    """
    Advanced PID controller with enhanced anti-windup and derivative filtering.
    
    Features:
    - Back-calculation anti-windup: Integrator corrected by K_aw * (clamped - raw)
    - Derivative-on-error: Low-pass filtered d(error)/dt for better setpoint tracking
    - Numerical stability: Prevents division by zero, handles edge cases
    - Velocity estimation: Exponential moving average of measurement derivative
    - Epsilon deadband: Prevents integral accumulation for very small errors
    """
    
    def __init__(self, Kp: float = 1.0, Ki: float = 0.0, Kd: float = 0.0,
                 output_limit: float = 1.0, derivative_alpha: float = 0.3,
                 anti_windup_gain: float = 1.0, velocity_alpha: float = 0.2,
                 integral_epsilon: float = 0.001,
                 integral_limit: float = np.inf):
        """
        Initialize PID controller.
        
        Args:
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            output_limit: Maximum absolute value for PID output (saturation)
            derivative_alpha: Low-pass filter coefficient for derivative (0-1, higher = less filtering)
                            Typical values: 0.2 = heavy filtering, 0.4 = moderate, 0.6 = light
            anti_windup_gain: Back-calculation anti-windup gain (typically 1.0)
            velocity_alpha: Exponential moving average coefficient for velocity estimation
            integral_epsilon: Deadband threshold for integral accumulation (meters)
                            Integral only accumulates when abs(error) >= epsilon
                            Prevents micro-oscillations from causing jerky behavior
                            Typical values: 0.0005-0.002 m (0.5-2 mm)
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.output_limit = output_limit
        self.derivative_alpha = derivative_alpha
        self.anti_windup_gain = anti_windup_gain
        self.velocity_alpha = velocity_alpha
        self.integral_epsilon = integral_epsilon
        self.integral_limit = integral_limit
        
        # Internal state
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = 0.0
        self.filtered_derivative = 0.0
        self.estimated_velocity = 0.0  # Filtered velocity estimate
        self.last_saturated = False
        self.pause_integral = False  # External pause flag (e.g., during bias estimation)
        
    def update(self, error: float, measurement: float, dt: float, pause_integral: bool = False) -> float:
        """
        Update PID controller and return control output.
        
        Args:
            error: Current error (setpoint - measurement)
            measurement: Current measurement value
            dt: Time step in seconds
            
        Returns:
            Control output (clamped to output_limit)
        """
        if dt <= 0:
            dt = 0.001  # Prevent division by zero
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term with epsilon deadband
        # Only accumulate integral when error is meaningfully non-zero
        # This prevents micro-oscillations from causing jerky behavior due to Ki buildup
        if not pause_integral and abs(error) >= self.integral_epsilon:
            self.integral += error * dt
        # If error is within deadband, integral remains unchanged (no accumulation)
        # Clamp integral to avoid excessive windup
        if np.isfinite(self.integral_limit):
            self.integral = float(np.clip(self.integral, -self.integral_limit, self.integral_limit))
        
        I = self.Ki * self.integral
        
        # Derivative term with filtering (derivative-on-error)
        raw_error_derivative = (error - self.prev_error) / dt
        
        # Low-pass filter on error derivative to reduce noise
        self.filtered_derivative = (self.derivative_alpha * raw_error_derivative + 
                                   (1 - self.derivative_alpha) * self.filtered_derivative)
        
        D = self.Kd * self.filtered_derivative
        
        # Compute raw output
        output_raw = P + I + D
        
        # Saturate output
        output_clamped = np.clip(output_raw, -self.output_limit, self.output_limit)
        self.last_saturated = bool(abs(output_clamped - output_raw) > 1e-9)
        
        # Back-calculation anti-windup: correct integrator based on saturation
        if self.Ki > 1e-6:  # Only if integral term is active
            windup_correction = self.anti_windup_gain * (output_clamped - output_raw) / self.Ki
            self.integral += windup_correction
        
        # Update velocity estimator (exponential moving average of measurement derivative)
        raw_velocity = (measurement - self.prev_measurement) / dt
        self.estimated_velocity = (self.velocity_alpha * raw_velocity + 
                                   (1 - self.velocity_alpha) * self.estimated_velocity)
        
        # Update state
        self.prev_error = error
        self.prev_measurement = measurement
        
        return output_clamped
    
    def get_velocity_estimate(self) -> float:
        """Get filtered velocity estimate."""
        return self.estimated_velocity
    
    def reset(self):
        """Reset internal state (integral, derivatives, velocity)."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = 0.0
        self.filtered_derivative = 0.0
        self.estimated_velocity = 0.0
        self.last_saturated = False
    
    def set_gains(self, Kp: float, Ki: float, Kd: float):
        """Update PID gains."""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def set_integral_limit(self, limit: float):
        """Set absolute clamp for the integral accumulator (anti-windup guard)."""
        self.integral_limit = max(0.0, float(limit)) if np.isfinite(limit) else np.inf

    def is_saturated(self) -> bool:
        """Return True if the last output was clamped to the output limits."""
        return self.last_saturated


class ControlState:
    """
    Manages control state: gains, target position, and runtime parameters.
    
    TUNING GUIDE FOR FASTER RESPONSE:
    1. Increase Kp for faster initial reaction (but too high causes oscillation)
    2. Increase Kd to dampen oscillations from higher Kp
    3. Increase tilt_gain for more aggressive platform movement
    4. Increase Ki carefully to eliminate steady-state error (but causes overshoot)
    5. Increase output_limit in ControlLoop if PID is saturating
    6. Increase derivative_alpha (0.3-0.5) for faster derivative response
    
    Current settings optimized for: Fast response with moderate stability
    """
    
    def __init__(self):
        # Target position (meters, platform coordinates)
        self.x_target = 0.0
        self.y_target = 0.0
        
        # Rate-limited target position (smooth setpoint changes)
        self.x_target_filtered = 0.0
        self.y_target_filtered = 0.0
        self.max_target_rate = 2.0  # m/s - maximum rate of setpoint change
        
        # PID gains (same for both x and y axes)
        # Tuned for faster response while maintaining stability
        self.Kp = 0.475  # Proportional gain - increased for faster reaction
        self.Ki = 0.00   # Integral gain - increased to eliminate steady-state error faster
        self.Kd = 1.2   # Derivative gain - increased for better damping at higher speeds
        
        # Control parameters
        self.max_tilt_angle = 5.0  # degrees (maximum platform tilt)
        self.tilt_gain = 0.4  # Scaling factor from PID output to tilt - increased for faster response
        self.normal_filter_alpha = 0.4  # Smoothing on normal vector (0=no smoothing, 1=instant)
        
        # Runtime flags
        self.running = False
        self.emergency_stop = False
        
        # Current state
        self.current_normal = np.array([0.0, 0.0, 1.0])
        self.filtered_normal = np.array([0.0, 0.0, 1.0])  # Smoothed normal vector
        self.ball_position = None  # (x, y) or None if not detected
        self.ball_is_centered = False  # True if ball is within centered tolerance
        self.last_update_time = None

        # Bias calibration and correction parameters
        self.bias_enabled = True
        # Epsilon used for both micro-error deadband and micro-bias ignore
        self.epsilon = 0.001  # meters
        # Centered margin: if error magnitude is below this, ball is considered centered
        # and control output is set to zero (prevents micro-adjustments)
        self.centered_tolerance = 0.003  # meters (3mm margin)
        self.steady_state_time = 1.0  # seconds window for bias detection
        self.delta_error_threshold = 0.001  # meters (error variation allowed in window)
        self.derivative_threshold = 0.003  # m/s (|d(error)/dt| below this means steady)
        self.max_bias_correction_rate = 0.01  # m/s (how fast to apply bias correction)
        self.bias_decay_rate = 0.2  # 1/s (decay toward zero when not steady)
        # Filtering and detection robustness for bias estimation
        self.bias_error_ema_alpha = 0.2  # (0-1) EMA on error for bias detection
        self.bias_sigma_threshold = 0.0006  # m, allowable std dev in steady window
        self.bias_min_window_samples = 10  # minimum samples in window to consider

        # Runtime bias state (for UI / diagnostics)
        self.x_bias_applied = 0.0
        self.y_bias_applied = 0.0
        self._last_x_target_for_bias = self.x_target
        self._last_y_target_for_bias = self.y_target


class BiasCalibratorAxis:
    """Detects and compensates steady-state bias for a single axis.

    Logic:
    - When error derivative is small and error variation is small over a time window,
      and the controller is not saturated, and |error| > epsilon, treat as steady-state.
    - Estimate bias as (measurement - setpoint) and gradually apply a correction with
      a max rate limit to avoid sudden jumps.
    - Pause integral accumulation during estimation windows and when micro-bias (< epsilon)
      would otherwise accumulate.
    - Decay applied bias toward zero when not in steady state.
    """

    def __init__(self,
                 epsilon: float,
                 steady_state_time: float,
                 delta_error_threshold: float,
                 derivative_threshold: float,
                 max_bias_correction_rate: float,
                 bias_decay_rate: float,
                 error_ema_alpha: float = 0.2,
                 sigma_threshold: float = 0.0006,
                 min_window_samples: int = 10):
        self.epsilon = float(epsilon)
        self.steady_state_time = float(steady_state_time)
        self.delta_error_threshold = float(delta_error_threshold)
        self.derivative_threshold = float(derivative_threshold)
        self.max_bias_correction_rate = float(max_bias_correction_rate)
        self.bias_decay_rate = float(bias_decay_rate)
        self.error_ema_alpha = float(np.clip(error_ema_alpha, 0.0, 1.0))
        self.sigma_threshold = float(sigma_threshold)
        self.min_window_samples = int(max(1, min_window_samples))

        self.history = deque()  # (timestamp, filtered_error)
        self.prev_error = 0.0
        self.error_ema = None
        self.applied_bias = 0.0
        self.last_saturated = False
        self._now = time.time

    def reset(self):
        self.history.clear()
        self.prev_error = 0.0
        self.error_ema = None
        self.applied_bias = 0.0
        self.last_saturated = False

    def set_last_saturated(self, saturated: bool):
        self.last_saturated = bool(saturated)

    def update(self, setpoint: float, measurement: float, dt: float, enabled: bool) -> Tuple[float, bool, bool]:
        """Update bias estimator.

        Returns: (applied_bias, pause_integral, estimating)
        """
        t = self._now()
        raw_error = setpoint - measurement
        # EMA filter for error to tolerate jitter
        if self.error_ema is None:
            self.error_ema = raw_error
        self.error_ema = (
            self.error_ema_alpha * raw_error + (1.0 - self.error_ema_alpha) * self.error_ema
        )
        e_filt = self.error_ema
        # Derivative on filtered error
        de_dt = (e_filt - self.prev_error) / max(dt, 1e-3)
        self.prev_error = e_filt

        # Maintain sliding window of errors
        self.history.append((t, e_filt))
        # Drop samples older than steady_state_time
        t_min = t - self.steady_state_time
        while self.history and self.history[0][0] < t_min:
            self.history.popleft()

        # Compute error variation over window
        err_vals = [e for (_, e) in self.history]
        delta_range = (max(err_vals) - min(err_vals)) if err_vals else float('inf')
        std_err = float(np.std(err_vals)) if err_vals else float('inf')
        window_time_ok = (self.history and (self.history[-1][0] - self.history[0][0]) >= self.steady_state_time)
        window_count_ok = (len(self.history) >= self.min_window_samples)
        window_ok = window_time_ok and window_count_ok

        # Conditions for steady-state bias detection
        steady = (
            enabled and
            (not self.last_saturated) and
            abs(de_dt) < self.derivative_threshold and
            (delta_range < self.delta_error_threshold or std_err < self.sigma_threshold) and
            abs(e_filt) > self.epsilon and
            window_ok
        )

        estimating = False
        if steady:
            estimating = True
            # Estimate bias as measurement - setpoint
            bias_estimate = measurement - setpoint
            # Rate-limit change toward estimate
            delta = bias_estimate - self.applied_bias
            max_step = self.max_bias_correction_rate * max(dt, 1e-3)
            step = float(np.clip(delta, -max_step, max_step))
            self.applied_bias += step
        else:
            # Decay bias toward zero when not steady
            decay = max(0.0, 1.0 - self.bias_decay_rate * max(dt, 1e-3))
            self.applied_bias *= decay

        # Epsilon interaction: don't apply micro-bias; also pause integral
        applied_bias = self.applied_bias if abs(self.applied_bias) >= self.epsilon else 0.0
        pause_integral = estimating or (abs(applied_bias) == 0.0 and abs(e_filt) < self.epsilon)

        return applied_bias, pause_integral, estimating


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
                 filter_alpha: float = 0.4):
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
            print(f"[TARGET] New target set: ({x_target:.4f}, {y_target:.4f}) m")
    
    def create_control_panel(self):
        """Create control panel window for parameter adjustment."""
        cv2.namedWindow(self.control_window)
        self.selected_param = 0  # Index of currently selected parameter
        self.param_names = ['Kp', 'Ki', 'Kd', 'TiltGain', 'MaxTilt', 'CenterTol']
        self.editing_mode = False
        self.edit_buffer = ""
        self.reset_callback = None  # Callback to reset PIDs when parameters change
    
    def set_reset_callback(self, callback):
        """Set callback function to reset PID controllers when parameters change."""
        self.reset_callback = callback
        
    def get_param_value(self, param_name: str) -> float:
        """Get current value of a parameter."""
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
        return 0.0
    
    def set_param_value(self, param_name: str, value: float):
        """Set value of a parameter and reset PID controllers to clear accumulated errors."""
        value = max(0.0, value)  # Ensure non-negative
        needs_reset = False  # Track if PID reset is needed
        
        if param_name == 'Kp':
            self.state.Kp = min(value, 50.0)
            needs_reset = True
        elif param_name == 'Ki':
            self.state.Ki = min(value, 20.0)
            needs_reset = True
        elif param_name == 'Kd':
            self.state.Kd = min(value, 10.0)
            needs_reset = True
        elif param_name == 'TiltGain':
            self.state.tilt_gain = min(value, 2.0)
            needs_reset = True
        elif param_name == 'MaxTilt':
            self.state.max_tilt_angle = min(value, 30.0)
            # MaxTilt doesn't need PID reset
        elif param_name == 'CenterTol':
            self.state.centered_tolerance = min(value, 0.05)  # Max 5cm
            # CenterTol doesn't need PID reset
        
        # Reset PID controllers if a control parameter was changed
        if needs_reset and self.reset_callback is not None:
            self.reset_callback()
            print(f"[RESET] PID integrals cleared after parameter change")
    
    def update_control_panel(self):
        """Update and display the control panel."""
        # Create blank image for control panel
        panel = np.zeros((400, 500, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark gray background
        
        # Title
        cv2.putText(panel, "PID Control Panel", (150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Instructions
        y_pos = 60
        instructions = [
            "W/S or UP/DOWN: Select parameter",
            "ENTER: Edit value",
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
            
            # Background highlight for selected parameter
            if is_selected:
                cv2.rectangle(panel, (5, y_pos - 20), (495, y_pos + 5), (80, 80, 80), -1)
            
            # Parameter name
            color = (0, 255, 255) if is_selected else (255, 255, 255)
            cv2.putText(panel, f"{param_name}:", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2 if is_selected else 1)
            
            # Parameter value (show edit buffer if editing this parameter)
            if is_selected and self.editing_mode:
                value_text = self.edit_buffer + "_"
                value_color = (0, 255, 0)  # Green when editing
            else:
                value_text = f"{value:.3f}"
                value_color = (255, 255, 255)
            
            cv2.putText(panel, value_text, (250, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, value_color, 2 if is_selected else 1)
            
            y_pos += 35
        
        # Status message
        y_pos += 10
        if self.editing_mode:
            status_msg = "EDITING MODE - Type value and press ENTER"
            status_color = (0, 255, 0)
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
            # Editing mode - capture number input
            if key == 13:  # ENTER - confirm edit
                try:
                    value = float(self.edit_buffer)
                    param_name = self.param_names[self.selected_param]
                    self.set_param_value(param_name, value)
                    print(f"[PARAM] {param_name} set to {value:.3f}")
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
            elif key == 13:  # ENTER - start editing
                param_name = self.param_names[self.selected_param]
                current_value = self.get_param_value(param_name)
                self.edit_buffer = f"{current_value:.3f}"
                self.editing_mode = True
                print(f"[PARAM] Editing {param_name} (current: {current_value:.3f})")
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
        
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        
        # Draw platform boundary circle (calibrated)
        center_x, center_y = self.plate_center_px
        
        # Draw crosshair at calibrated platform center (not frame center)
        cv2.line(overlay, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 255), 2)
        cv2.line(overlay, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 255), 2)
        cv2.circle(overlay, (center_x, center_y), 5, (0, 255, 255), -1)
        cv2.putText(overlay, "Origin", (center_x + 10, center_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw coordinate axes if calibrated
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
            
            # Draw tick marks along axes to show scale (every 5cm)
            tick_interval_m = 0.05  # 5cm ticks
            tick_length = 8
            max_distance_m = self.plate_radius_m
            
            # X-axis tick marks
            for sign in [1, -1]:
                distance_m = tick_interval_m
                while distance_m <= max_distance_m:
                    tick_px_along_axis = distance_m / self.pixel_to_meter_ratio
                    tick_center = (int(center_x + sign * self.x_axis[0] * tick_px_along_axis),
                                  int(center_y + sign * self.x_axis[1] * tick_px_along_axis))
                    # Perpendicular to x_axis is y_axis
                    tick_start = (int(tick_center[0] - self.y_axis[0] * tick_length),
                                 int(tick_center[1] - self.y_axis[1] * tick_length))
                    tick_end = (int(tick_center[0] + self.y_axis[0] * tick_length),
                               int(tick_center[1] + self.y_axis[1] * tick_length))
                    cv2.line(overlay, tick_start, tick_end, (0, 0, 200), 1)
                    distance_m += tick_interval_m
            
            # Y-axis tick marks
            for sign in [1, -1]:
                distance_m = tick_interval_m
                while distance_m <= max_distance_m:
                    tick_px_along_axis = distance_m / self.pixel_to_meter_ratio
                    tick_center = (int(center_x + sign * self.y_axis[0] * tick_px_along_axis),
                                  int(center_y + sign * self.y_axis[1] * tick_px_along_axis))
                    # Perpendicular to y_axis is x_axis
                    tick_start = (int(tick_center[0] - self.x_axis[0] * tick_length),
                                 int(tick_center[1] - self.x_axis[1] * tick_length))
                    tick_end = (int(tick_center[0] + self.x_axis[0] * tick_length),
                               int(tick_center[1] + self.x_axis[1] * tick_length))
                    cv2.line(overlay, tick_start, tick_end, (0, 200, 0), 1)
                    distance_m += tick_interval_m
        
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
                status_text = "CENTERED" if self.state.ball_is_centered else f"({x:.3f}, {y:.3f})m"
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
            tilt_x = -n[0] * tilt_scale
            tilt_y = -n[1] * tilt_scale
            
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
        cv2.putText(overlay, f"Target: ({self.state.x_target:.3f}, {self.state.y_target:.3f}) m",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        
        if self.state.ball_position is not None:
            x, y = self.state.ball_position
            ex = self.state.x_target - x
            ey = self.state.y_target - y
            error_mag = np.sqrt(ex**2 + ey**2)
            error_color = (0, 255, 0) if self.state.ball_is_centered else (255, 255, 0)
            status_str = " [CENTERED]" if self.state.ball_is_centered else ""
            cv2.putText(overlay, f"Error: {error_mag:.4f}m ({ex:.3f}, {ey:.3f}){status_str}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, error_color, 2)
            y_offset += 25
        
        n = self.state.current_normal
        tilt_angle = np.rad2deg(np.arccos(np.clip(n[2], -1.0, 1.0)))
        cv2.putText(overlay, f"Normal: ({n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f})",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        cv2.putText(overlay, f"Tilt: {tilt_angle:.1f} deg",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # PID gains
        y_offset += 30
        cv2.putText(
            overlay,
            f"Kp: {self.state.Kp:.1f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y_offset += 20
        cv2.putText(
            overlay,
            f"Ki: {self.state.Ki:.2f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y_offset += 20
        cv2.putText(
            overlay,
            f"Kd: {self.state.Kd:.1f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y_offset += 20
        cv2.putText(
            overlay,
            f"BiasEnabled: {'ON' if self.state.bias_enabled else 'OFF'}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 255, 200),
            1,
        )
        y_offset += 20
        cv2.putText(
            overlay,
            f"Bias: ({self.state.x_bias_applied:.3f}, {self.state.y_bias_applied:.3f}) m",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 255, 200),
            1,
        )
        y_offset += 20
        cv2.putText(
            overlay,
            f"CenterTol: {self.state.centered_tolerance*1000:.1f}mm",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 200, 255),
            1,
        )

        # Instructions
        y_offset = h - 60
        cv2.putText(
            overlay,
            "Click to set target | 'q' quit | 'r' reset | 'b' bias on/off",
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
                 servo_controller: ArduinoServoController):
        """
        Initialize control loop.
        
        Args:
            state: ControlState instance
            camera_manager: CameraManager instance
            normal_controller: NormalController instance
            ui_manager: UIManager instance
            servo_controller: ArduinoServoController instance
        """
        self.state = state
        self.camera_manager = camera_manager
        self.normal_controller = normal_controller
        self.ui_manager = ui_manager
        self.servo_controller = servo_controller
        
        # PID controllers (same gains for both X and Y)
        self.pid_x = PIDController(
            Kp=state.Kp, Ki=state.Ki, Kd=state.Kd,
            output_limit=20.0,  # Increased output limit for faster response
            derivative_alpha=0.3,  # Moderate filtering for responsive derivative
            anti_windup_gain=1.0,  # Back-calculation anti-windup
            velocity_alpha=0.2,  # Velocity estimation filtering
            integral_epsilon=self.state.epsilon  # Deadband for integral (prevents micro-oscillation buildup)
        )
        self.pid_y = PIDController(
            Kp=state.Kp, Ki=state.Ki, Kd=state.Kd,
            output_limit=20.0,  # Increased output limit for faster response
            derivative_alpha=0.3,  # Moderate filtering for responsive derivative
            anti_windup_gain=1.0,  # Back-calculation anti-windup
            velocity_alpha=0.2,  # Velocity estimation filtering
            integral_epsilon=self.state.epsilon  # Deadband for integral (prevents micro-oscillation buildup)
        )
        # Set conservative integral clamp to complement back-calculation anti-windup
        self.pid_x.set_integral_limit(1.0)
        self.pid_y.set_integral_limit(1.0)

        # Bias calibrators for X and Y axes
        self.bias_x = BiasCalibratorAxis(
            epsilon=self.state.epsilon,
            steady_state_time=self.state.steady_state_time,
            delta_error_threshold=self.state.delta_error_threshold,
            derivative_threshold=self.state.derivative_threshold,
            max_bias_correction_rate=self.state.max_bias_correction_rate,
            bias_decay_rate=self.state.bias_decay_rate,
            error_ema_alpha=self.state.bias_error_ema_alpha,
            sigma_threshold=self.state.bias_sigma_threshold,
            min_window_samples=self.state.bias_min_window_samples,
        )
        self.bias_y = BiasCalibratorAxis(
            epsilon=self.state.epsilon,
            steady_state_time=self.state.steady_state_time,
            delta_error_threshold=self.state.delta_error_threshold,
            derivative_threshold=self.state.derivative_threshold,
            max_bias_correction_rate=self.state.max_bias_correction_rate,
            bias_decay_rate=self.state.bias_decay_rate,
            error_ema_alpha=self.state.bias_error_ema_alpha,
            sigma_threshold=self.state.bias_sigma_threshold,
            min_window_samples=self.state.bias_min_window_samples,
        )
        # Bias estimation debug helpers
        self._last_bias_estimating = False
        self._last_bias_print = 0.0
        
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
        print("  - Press 'q' to quit, 'r' to reset PID integrals and bias")
        print("  - Press 'b' to toggle bias compensation ON/OFF")
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
            # Also reset bias estimators when core control parameters change
            self.bias_x.reset()
            self.bias_y.reset()
            self.state.x_bias_applied = 0.0
            self.state.y_bias_applied = 0.0
        self.ui_manager.set_reset_callback(reset_pids)
        
        self.state.running = True
        self.state.last_update_time = time.time()
        
        # Initialize filtered setpoint to current target
        self.state.x_target_filtered = self.state.x_target
        self.state.y_target_filtered = self.state.y_target
        
        # Control loop
        while self.state.running and not self.state.emergency_stop:
            loop_start = time.time()
            
            # Update camera and ball detection
            self.camera_manager.update()
            
            # Update control panel display
            self.ui_manager.update_control_panel()
            
            # Update PID gains and normal controller parameters
            self.pid_x.set_gains(self.state.Kp, self.state.Ki, self.state.Kd)
            self.pid_y.set_gains(self.state.Kp, self.state.Ki, self.state.Kd)
            self.normal_controller.set_tilt_gain(self.state.tilt_gain)
            self.normal_controller.set_max_tilt_angle(self.state.max_tilt_angle)
            self.normal_controller.set_filter_alpha(self.state.normal_filter_alpha)
            
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
            
            # Reset bias calibrators if target changed significantly
            if (abs(self.state.x_target - self.state._last_x_target_for_bias) > self.state.epsilon or
                abs(self.state.y_target - self.state._last_y_target_for_bias) > self.state.epsilon):
                self.bias_x.reset()
                self.bias_y.reset()
                self.state._last_x_target_for_bias = self.state.x_target
                self.state._last_y_target_for_bias = self.state.y_target

            # Control update (only if ball detected)
            if ball_pos is not None:
                x, y = ball_pos
                
                # Bias estimation and compensation
                bx, pause_ix, estimating_x = self.bias_x.update(
                    setpoint=self.state.x_target_filtered,
                    measurement=x,
                    dt=dt,
                    enabled=self.state.bias_enabled,
                )
                by, pause_iy, estimating_y = self.bias_y.update(
                    setpoint=self.state.y_target_filtered,
                    measurement=y,
                    dt=dt,
                    enabled=self.state.bias_enabled,
                )
                x_corr = x - (bx if self.state.bias_enabled else 0.0)
                y_corr = y - (by if self.state.bias_enabled else 0.0)
                self.state.x_bias_applied = (bx if self.state.bias_enabled else 0.0)
                self.state.y_bias_applied = (by if self.state.bias_enabled else 0.0)

                # Lightweight diagnostic to confirm bias estimator is active
                estimating_any = estimating_x or estimating_y
                if estimating_any and (not self._last_bias_estimating) and (current_time - self._last_bias_print > 1.0):
                    print(f"[BIAS] Estimating... current bias approx (x={bx:+.4f}, y={by:+.4f}) m")
                    self._last_bias_print = current_time
                self._last_bias_estimating = estimating_any

                # Compute errors using rate-limited setpoint and corrected measurement
                ex = self.state.x_target_filtered - x_corr
                ey = self.state.y_target_filtered - y_corr
                
                # Check if ball is within centered tolerance
                error_magnitude = np.sqrt(ex**2 + ey**2)
                is_centered = error_magnitude < self.state.centered_tolerance
                self.state.ball_is_centered = is_centered
                
                # Update PID controllers with optional integral pause during bias estimation
                # If ball is centered (within tolerance), output zero and don't update PID
                if is_centered:
                    # Ball is centered - set outputs to zero and pause integral
                    ux = 0.0
                    uy = 0.0
                    # Still update PIDs with pause_integral=True to maintain state tracking
                    # but ignore the output
                    _ = self.pid_x.update(ex, x_corr, dt, pause_integral=True)
                    _ = self.pid_y.update(ey, y_corr, dt, pause_integral=True)
                else:
                    # Ball is not centered - normal PID control
                    ux = self.pid_x.update(ex, x_corr, dt, pause_integral=pause_ix)
                    uy = self.pid_y.update(ey, y_corr, dt, pause_integral=pause_iy)

                # Update calibrators with saturation info for next iteration
                self.bias_x.set_last_saturated(self.pid_x.is_saturated())
                self.bias_y.set_last_saturated(self.pid_y.is_saturated())
                
                # Compute desired normal (includes filtering)
                n = self.normal_controller.compute_normal(ux, uy)
                self.state.current_normal = n
                self.state.filtered_normal = self.normal_controller.filtered_normal
                
                # Send to hardware via Arduino servo controller
                self.servo_controller.set_normal(n[0], n[1], n[2])
                
                # Print diagnostics (throttled to avoid console spam)
                # Only print every 10 iterations (~10 Hz instead of 100 Hz)
                if hasattr(self, '_print_counter'):
                    self._print_counter += 1
                else:
                    self._print_counter = 0
                
                if self._print_counter % 10 == 0:
                    tilt_angle = np.rad2deg(np.arccos(np.clip(n[2], -1.0, 1.0)))
                    centered_str = " [CENTERED]" if is_centered else ""
                    print(
                        f"[CONTROL] Pos: ({x:+.4f}, {y:+.4f}) m | "
                        f"Error: ({ex:+.4f}, {ey:+.4f}) m{centered_str} | "
                        f"PID: ({ux:+.2f}, {uy:+.2f}) | "
                        f"Bias: ({self.state.x_bias_applied:+.4f}, {self.state.y_bias_applied:+.4f}) m | "
                        f"Normal: ({n[0]:+.3f}, {n[1]:+.3f}, {n[2]:+.3f}) | "
                        f"Tilt: {tilt_angle:.1f}°"
                    )
            
            # Update display
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
                        print("[RESET] Resetting PID integrals and bias")
                        self.pid_x.reset()
                        self.pid_y.reset()
                        self.bias_x.reset()
                        self.bias_y.reset()
                        self.state.x_bias_applied = 0.0
                        self.state.y_bias_applied = 0.0
                    elif key == ord('b'):
                        self.state.bias_enabled = not self.state.bias_enabled
                        print(f"[BIAS] Bias compensation {'ENABLED' if self.state.bias_enabled else 'DISABLED'}")
        
        # Cleanup: set platform to flat
        print("[CLEANUP] Setting platform to flat")
        self.servo_controller.set_normal(0.0, 0.0, 1.0)
        self.camera_manager.close()
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
        """
    )
    parser.add_argument(
        'calib_file',
        nargs='?',
        default=None,
        help='Path to camera calibration file (.npz). If not provided, camera calibration is disabled.'
    )
    args = parser.parse_args()
    
    print("="*70)
    print("3D Stewart Platform PID Controller with Ball Detection")
    print("="*70)
    
    if args.calib_file:
        print(f"[CONFIG] Camera calibration file: {args.calib_file}")
    else:
        print("[CONFIG] Camera calibration: DISABLED")
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
    control_loop = ControlLoop(state, camera_manager, normal_controller, ui_manager, servo_controller)
    
    try:
        control_loop.run()
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user")
        state.emergency_stop = True
        servo_controller.set_normal(0.0, 0.0, 1.0)
        camera_manager.close()
        servo_controller.close()
    except Exception as e:
        print(f"[ERROR] Control loop failed: {e}")
        import traceback
        traceback.print_exc()
        servo_controller.set_normal(0.0, 0.0, 1.0)
        camera_manager.close()
        servo_controller.close()


if __name__ == "__main__":
    main()
