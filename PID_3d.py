#!/usr/bin/env python3
"""
3D Stewart Platform PID Controller
Real-time 2D PID control for ball balancing on a 3-motor Stewart platform.

This controller integrates:
- Ball detection (BallDetector from ball_detection.py) for real-time vision
- PID control for X and Y axes with live tuning
- Normal vector computation from PID outputs
- Arduino servo control (based on arduino_controller.py)
- Inverse kinematics (from inverseKinematics.py) for servo angle calculation

Features:
- Reads ball position (x, y) in meters from camera with calibration
- Uses independent PID controllers for x and y axes
- Maps PID outputs to a desired platform normal vector
- Provides live tuning via OpenCV trackbars
- Allows target selection via mouse clicks on camera feed
- Sends computed servo angles to Arduino hardware

Hardware Interface:
- Connects to Arduino via serial (auto-detects port)
- Uses StewartPlatform inverse kinematics to compute servo angles
- Falls back to simulation mode if Arduino not connected

Usage:
    python PID_3d.py
    
Controls:
    - Click on camera window to set target position
    - Use trackbars in 'PID Tuning' window to adjust gains
    - Press 'q' to quit, 'r' to reset PID integrals
"""

import cv2
import numpy as np
import time
import sys
import serial
import serial.tools.list_ports
from typing import Optional, Tuple

# Import ball detection and inverse kinematics
from ball_detection import BallDetector
from inverseKinematics import StewartPlatform


class PIDController:
    """
    Generic PID controller with anti-windup and derivative filtering.
    
    Features:
    - Anti-windup: Integral term clamping to prevent saturation
    - Derivative filtering: Low-pass filter on derivative to reduce noise
    - Derivative-on-measurement: Uses measurement instead of error for smoother response
    """
    
    def __init__(self, Kp: float = 1.0, Ki: float = 0.0, Kd: float = 0.0,
                 integral_limit: float = 10.0, output_limit: float = 1.0,
                 derivative_alpha: float = 0.1):
        """
        Initialize PID controller.
        
        Args:
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            integral_limit: Maximum absolute value for integral term (anti-windup)
            output_limit: Maximum absolute value for PID output (saturation)
            derivative_alpha: Low-pass filter coefficient for derivative (0-1, lower = more filtering)
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.derivative_alpha = derivative_alpha
        
        # Internal state
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = 0.0
        self.filtered_derivative = 0.0
        
    def update(self, error: float, measurement: float, dt: float) -> float:
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
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        I = self.Ki * self.integral
        
        # Derivative term with filtering (derivative-on-measurement)
        # Using measurement derivative reduces kick on setpoint changes
        raw_derivative = (measurement - self.prev_measurement) / dt if dt > 0 else 0.0
        
        # Low-pass filter on derivative to reduce noise
        self.filtered_derivative = (self.derivative_alpha * raw_derivative + 
                                   (1 - self.derivative_alpha) * self.filtered_derivative)
        
        # Derivative term uses negative of filtered derivative (since we want to oppose changes)
        D = -self.Kd * self.filtered_derivative
        
        # Update state
        self.prev_error = error
        self.prev_measurement = measurement
        
        # Compute output
        output = P + I + D
        
        # Saturate output
        output = np.clip(output, -self.output_limit, self.output_limit)
        
        return output
    
    def reset(self):
        """Reset internal state (integral, derivatives)."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = 0.0
        self.filtered_derivative = 0.0
    
    def set_gains(self, Kp: float, Ki: float, Kd: float):
        """Update PID gains."""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd


class ControlState:
    """Manages control state: gains, target position, and runtime parameters."""
    
    def __init__(self):
        # Target position (meters, platform coordinates)
        self.x_target = 0.0
        self.y_target = 0.0
        
        # PID gains (same for both x and y axes)
        # Conservative starting values for stability
        self.Kp = 5.0   # Proportional gain
        self.Ki = 0.1   # Integral gain (low to prevent windup)
        self.Kd = 1.0   # Derivative gain (damping)
        
        # Control parameters
        self.control_frequency = 100.0  # Hz
        self.max_tilt_angle = 15.0  # degrees (maximum platform tilt)
        self.tilt_gain = 0.5  # Scaling factor from PID output to tilt
        
        # Runtime flags
        self.running = False
        self.emergency_stop = False
        
        # Current state
        self.current_normal = np.array([0.0, 0.0, 1.0])
        self.ball_position = None  # (x, y) or None if not detected
        self.last_update_time = None


class NormalController:
    """
    Maps PID control outputs to a desired platform normal vector.
    
    Sign Convention:
    - Positive PID output ux means "tilt platform so ball accelerates toward +x"
    - This corresponds to tilting the platform such that the normal has negative nx
    - For small angles: nx ≈ -k * ux, ny ≈ -k * uy
    - Normal is always normalized to unit length with positive nz
    """
    
    def __init__(self, tilt_gain: float = 0.5, max_tilt_angle_deg: float = 15.0):
        """
        Initialize normal controller.
        
        Args:
            tilt_gain: Scaling factor from PID output to tilt (radians per unit PID output)
            max_tilt_angle_deg: Maximum platform tilt angle in degrees
        """
        self.tilt_gain = tilt_gain
        self.max_tilt_angle_rad = np.deg2rad(max_tilt_angle_deg)
        
    def compute_normal(self, ux: float, uy: float) -> np.ndarray:
        """
        Compute desired platform normal from PID outputs.
        
        Args:
            ux: PID output for x-axis (control demand)
            uy: PID output for y-axis (control demand)
            
        Returns:
            Unit normal vector [nx, ny, nz] where ||n|| = 1 and nz > 0
            
        Algorithm:
        1. Scale PID outputs by tilt_gain to get desired tilt angles
        2. For small angles, construct unnormalized normal: n_raw = (-ux, -uy, 1.0)
           (negative because positive ux should tilt platform left to move ball right)
        3. Normalize to unit length
        4. Optionally clamp to maximum tilt angle
        """
        # Scale PID outputs to get desired tilt components
        # Negative sign: positive ux (ball too far right) → tilt left (negative nx)
        nx_raw = -self.tilt_gain * ux
        ny_raw = -self.tilt_gain * uy
        
        # Clamp to maximum tilt angle
        tilt_magnitude = np.sqrt(nx_raw**2 + ny_raw**2)
        if tilt_magnitude > np.tan(self.max_tilt_angle_rad):
            scale = np.tan(self.max_tilt_angle_rad) / tilt_magnitude
            nx_raw *= scale
            ny_raw *= scale
        
        # Construct unnormalized normal vector
        # For small tilts: n ≈ (-slope_x, -slope_y, 1)
        nz_raw = 1.0
        n_raw = np.array([nx_raw, ny_raw, nz_raw])
        
        # Normalize to unit length
        n_norm = np.linalg.norm(n_raw)
        if n_norm > 1e-6:
            n = n_raw / n_norm
        else:
            n = np.array([0.0, 0.0, 1.0])  # Flat platform
        
        # Ensure nz is positive (platform facing up)
        if n[2] < 0:
            n = -n
        
        return n
    
    def set_tilt_gain(self, gain: float):
        """Update tilt gain."""
        self.tilt_gain = gain
    
    def set_max_tilt_angle(self, angle_deg: float):
        """Update maximum tilt angle."""
        self.max_tilt_angle_rad = np.deg2rad(angle_deg)


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
            
            # Read response
            time.sleep(0.01)  # Small delay for Arduino to respond
            if self.ser.in_waiting:
                response = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if response.startswith('OK:'):
                    return True
                elif response.startswith('ERR:'):
                    print(f"[ARDUINO] Error: {response}")
                    return False
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
    
    def __init__(self, camera_id=0, config_file="config.json", calib_file="cal/camera_calib.npz"):
        """
        Initialize camera and ball detector.
        
        Args:
            camera_id: Camera device ID (default: 0)
            config_file: Path to config.json
            calib_file: Path to camera calibration file
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
                 plate_radius_m: float = 0.15, frame_width: int = 640, frame_height: int = 480):
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
    
    def create_trackbars(self):
        """Create OpenCV trackbars for PID tuning."""
        cv2.namedWindow(self.control_window)
        
        # PID gains (trackbars use integers, scale appropriately)
        cv2.createTrackbar("Kp", self.control_window, 
                          max(0, min(500, int(self.state.Kp * 10))), 500, self.on_trackbar)
        cv2.createTrackbar("Ki", self.control_window, 
                          max(0, min(200, int(self.state.Ki * 100))), 200, self.on_trackbar)
        cv2.createTrackbar("Kd", self.control_window, 
                          max(0, min(100, int(self.state.Kd * 10))), 100, self.on_trackbar)
        
        # Control parameters
        cv2.createTrackbar("TiltGain", self.control_window, 
                          max(0, min(200, int(self.state.tilt_gain * 100))), 200, self.on_trackbar)
        cv2.createTrackbar("MaxTilt", self.control_window, 
                          max(0, min(30, int(self.state.max_tilt_angle))), 30, self.on_trackbar)
    
    def on_trackbar(self, val):
        """Trackbar callback (OpenCV requirement - not used directly)."""
        self.update_from_trackbars()
    
    def update_from_trackbars(self):
        """Read trackbar values and update state."""
        self.state.Kp = cv2.getTrackbarPos("Kp", self.control_window) / 10.0
        self.state.Ki = cv2.getTrackbarPos("Ki", self.control_window) / 100.0
        self.state.Kd = cv2.getTrackbarPos("Kd", self.control_window) / 10.0
        
        self.state.tilt_gain = cv2.getTrackbarPos("TiltGain", self.control_window) / 100.0
        self.state.max_tilt_angle = cv2.getTrackbarPos("MaxTilt", self.control_window)
    
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
            cv2.drawMarker(overlay, (target_x_px, target_y_px), (0, 255, 0), 
                          cv2.MARKER_CROSS, 20, 2)
            cv2.putText(overlay, "Target", (target_x_px + 5, target_y_px - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw ball position if available (using calibrated conversion)
        if self.state.ball_position is not None:
            x, y = self.state.ball_position
            ball_x_px, ball_y_px = self.platform_to_pixel(x, y)
            if 0 <= ball_x_px < w and 0 <= ball_y_px < h:
                cv2.circle(overlay, (ball_x_px, ball_y_px), 10, (255, 0, 255), 2)
                cv2.putText(overlay, f"Ball: ({x:.3f}, {y:.3f})m", 
                           (ball_x_px + 15, ball_y_px), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 0, 255), 2)
        
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
            cv2.putText(overlay, f"Error: ({ex:.3f}, {ey:.3f}) m",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
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
        cv2.putText(overlay, f"Kp: {self.state.Kp:.1f}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(overlay, f"Ki: {self.state.Ki:.2f}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(overlay, f"Kd: {self.state.Kd:.1f}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        y_offset = h - 60
        cv2.putText(overlay, "Click to set target | 'q' to quit | 'r' to reset",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
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
            integral_limit=5.0, output_limit=10.0
        )
        self.pid_y = PIDController(
            Kp=state.Kp, Ki=state.Ki, Kd=state.Kd,
            integral_limit=5.0, output_limit=10.0
        )
        
    def run(self):
        """Run main control loop."""
        print("="*70)
        print("3D Stewart Platform PID Controller")
        print("="*70)
        print("Controls:")
        print("  - Click on camera feed to set target position")
        print("  - Use trackbars in 'PID Tuning' window to adjust gains")
        print("  - Press 'q' to quit, 'r' to reset PID integrals")
        print("="*70)
        
        # Initialize UI
        cv2.namedWindow(self.ui_manager.window_name)
        cv2.setMouseCallback(self.ui_manager.window_name, self.ui_manager.mouse_callback)
        self.ui_manager.create_trackbars()
        
        self.state.running = True
        self.state.last_update_time = time.time()
        
        # Control loop
        target_dt = 1.0 / self.state.control_frequency
        first_iteration = True
        
        while self.state.running and not self.state.emergency_stop:
            loop_start = time.time()
            
            # Update camera and ball detection
            self.camera_manager.update()
            
            # Update PID gains from trackbars (same for both X and Y)
            self.ui_manager.update_from_trackbars()
            self.pid_x.set_gains(self.state.Kp, self.state.Ki, self.state.Kd)
            self.pid_y.set_gains(self.state.Kp, self.state.Ki, self.state.Kd)
            self.normal_controller.set_tilt_gain(self.state.tilt_gain)
            self.normal_controller.set_max_tilt_angle(self.state.max_tilt_angle)
            
            # Get ball position
            ball_pos = self.camera_manager.get_ball_position()
            self.state.ball_position = ball_pos
            
            # Calculate dt (handle first iteration)
            current_time = time.time()
            if first_iteration:
                dt = target_dt  # Use target dt for first iteration
                first_iteration = False
            else:
                dt = current_time - self.state.last_update_time
            
            self.state.last_update_time = current_time
            
            # Clamp dt to reasonable range (handle long pauses)
            dt = np.clip(dt, 0.001, 0.1)
            
            # Control update (only if ball detected)
            if ball_pos is not None:
                x, y = ball_pos
                
                # Compute errors
                ex = self.state.x_target - x
                ey = self.state.y_target - y
                
                # Update PID controllers
                ux = self.pid_x.update(ex, x, dt)
                uy = self.pid_y.update(ey, y, dt)
                
                # Compute desired normal
                n = self.normal_controller.compute_normal(ux, uy)
                self.state.current_normal = n
                
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
                    print(f"[CONTROL] Pos: ({x:+.4f}, {y:+.4f}) m | "
                          f"Error: ({ex:+.4f}, {ey:+.4f}) m | "
                          f"PID: ({ux:+.2f}, {uy:+.2f}) | "
                          f"Normal: ({n[0]:+.3f}, {n[1]:+.3f}, {n[2]:+.3f}) | "
                          f"Tilt: {tilt_angle:.1f}°")
            
            # Update display
            frame = self.camera_manager.get_camera_frame()
            if frame is not None:
                overlay = self.ui_manager.draw_overlay(frame)
                if overlay is not None:
                    cv2.imshow(self.ui_manager.window_name, overlay)
            
            # Handle keyboard input (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[STOP] Emergency stop requested")
                self.state.emergency_stop = True
                break
            elif key == ord('r'):
                print("[RESET] Resetting PID integrals")
                self.pid_x.reset()
                self.pid_y.reset()
            
            # Maintain control frequency
            elapsed = time.time() - loop_start
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Cleanup: set platform to flat
        print("[CLEANUP] Setting platform to flat")
        self.servo_controller.set_normal(0.0, 0.0, 1.0)
        self.camera_manager.close()
        cv2.destroyAllWindows()
        self.state.running = False

def main():
    """Main entry point."""
    print("="*70)
    print("3D Stewart Platform PID Controller with Ball Detection")
    print("="*70)
    
    # Initialize components
    state = ControlState()
    
    # Initialize hardware (Arduino servo controller)
    print("[INIT] Connecting to Arduino...")
    servo_controller = ArduinoServoController()
    
    # Initialize camera and ball detection
    print("[INIT] Initializing camera and ball detector...")
    camera_manager = CameraManager(camera_id=0)
    
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
        frame_width=640,
        frame_height=480
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
