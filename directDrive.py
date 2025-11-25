#!/usr/bin/env python3
"""
Direct-Drive Stewart Platform PID Controller
Real-time direct motor angle control for ball balancing on a 3-motor Stewart platform.

This controller replaces both PID_3d.py and inverseKinematics.py by:
- Directly computing motor angles from ball position error projections
- Using 3 independent PID controllers (one per motor axis)
- Projecting error vector onto motor directions at 0°, 120°, 240°
- No inverse kinematics - direct motor space control

Motor Geometry:
- Motor A: 0° (along +X axis)
- Motor B: 120° (CCW from +X)
- Motor C: 240° (CCW from +X)

Control Flow:
Error Vector → Project onto Motor Axes → 3 PIDs → Motor Angles → Arduino

Usage:
    python directDrive.py                        # Run with default settings
    python directDrive.py cal/camera_calib.npz   # Run with camera calibration
    python directDrive.py --dt 0.1               # Run at fixed 10Hz
    
Controls:
    - Click on camera window to set target position
    - Press 'q' to quit, 'r' to reset PID integrals
"""

import cv2
import numpy as np
import time
import argparse
import serial
import serial.tools.list_ports
from typing import Optional, Tuple
import csv
import os
from datetime import datetime

# Import ball detection
from ball_detection import BallDetector


# ============================================================================
# PID Controller
# ============================================================================

class PIDController:
    """
    Simple PID controller with anti-windup.
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
        
    def update(self, error: float, dt: float) -> float:
        """
        Update PID controller and return control output.
        
        Args:
            error: Current error
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


# ============================================================================
# Motor Geometry
# ============================================================================

class MotorGeometry:
    """
    Motor geometry calculations for direct-drive control.
    
    Motors are arranged at 0°, 120°, 240° (CCW from +X axis).
    """
    
    @staticmethod
    def get_motor_unit_vectors():
        """
        Return unit vectors for motors A, B, C at 0°, 120°, 240°.
        
        Motor A is at 0° along the +X axis: [cos(0°), sin(0°)] = [1, 0]
        Motor B is at 120° CCW from +X: [cos(120°), sin(120°)] = [-0.5, √3/2]
        Motor C is at 240° CCW from +X: [cos(240°), sin(240°)] = [-0.5, -√3/2]
        
        Returns:
            Tuple of (motor_A, motor_B, motor_C) as numpy arrays
            Each vector is [cos(angle), sin(angle)] for the corresponding motor angle
        """
        # Motor A at 0°: [cos(0°), sin(0°)] = [1, 0] (along +X axis)
        motor_A = np.array([1.0, 0.0])
        
        # Motor B at 120°: [cos(120°), sin(120°)] = [-0.5, √3/2]
        motor_B = np.array([np.cos(np.radians(120)), np.sin(np.radians(120))])
        
        # Motor C at 240°: [cos(240°), sin(240°)] = [-0.5, -√3/2]
        motor_C = np.array([np.cos(np.radians(240)), np.sin(np.radians(240))])
        
        return motor_A, motor_B, motor_C
    
    @staticmethod
    def project_error(error_vector: np.ndarray, motor_vectors: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Project error vector onto each motor axis.
        
        Motor vectors are [cos(angle), sin(angle)] where angle is measured CCW from +X axis.
        For error vector [ex, ey], projection is: [ex, ey] dot [cos(angle), sin(angle)]
        
        Args:
            error_vector: 2D error vector [ex, ey]
            motor_vectors: Tuple of (motor_A, motor_B, motor_C) unit vectors
                          Each vector is [cos(angle), sin(angle)]
            
        Returns:
            Array of projections [proj_A, proj_B, proj_C]
        """
        motor_A, motor_B, motor_C = motor_vectors
        proj_A = np.dot(error_vector, motor_A)
        proj_B = np.dot(error_vector, motor_B)
        proj_C = np.dot(error_vector, motor_C)
        return np.array([proj_A, proj_B, proj_C])


# ============================================================================
# Direct Drive Controller
# ============================================================================

class DirectDriveController:
    """
    Direct-drive controller that computes motor angles from error projections.
    
    Uses 3 independent PID controllers, one per motor axis.
    """
    
    def __init__(self, Kp: float = 10.0, Ki: float = 5.0, Kd: float = 8.0,
                 Kp_A: float = None, Ki_A: float = None, Kd_A: float = None,
                 Kp_B: float = None, Ki_B: float = None, Kd_B: float = None,
                 Kp_C: float = None, Ki_C: float = None, Kd_C: float = None,
                 output_limit: float = 25.0, neutral_angle: float = 15.5):
        """
        Initialize direct-drive controller.
        
        Args:
            Kp, Ki, Kd: Default gains (used if individual motor gains not specified)
            Kp_A, Ki_A, Kd_A: Individual gains for Motor A (overrides defaults)
            Kp_B, Ki_B, Kd_B: Individual gains for Motor B (overrides defaults)
            Kp_C, Ki_C, Kd_C: Individual gains for Motor C (overrides defaults)
            output_limit: Maximum PID output magnitude in degrees
            neutral_angle: Base angle offset (accounts for servo zero position)
        """
        self.neutral_angle = neutral_angle
        self.output_limit = output_limit
        
        # Get motor unit vectors
        self.motor_vectors = MotorGeometry.get_motor_unit_vectors()
        
        # Create 3 independent PID controllers with individual gains
        # Note: Gains are scaled for meter inputs (projections are in meters)
        # With Kp=10, a 0.025m projection gives 0.25 degrees output
        # With Kp=10, a 0.1m projection gives 1.0 degree output
        Kp_A_final = Kp_A if Kp_A is not None else Kp
        Ki_A_final = Ki_A if Ki_A is not None else Ki
        Kd_A_final = Kd_A if Kd_A is not None else Kd
        
        Kp_B_final = Kp_B if Kp_B is not None else Kp
        Ki_B_final = Ki_B if Ki_B is not None else Ki
        Kd_B_final = Kd_B if Kd_B is not None else Kd
        
        Kp_C_final = Kp_C if Kp_C is not None else Kp
        Ki_C_final = Ki_C if Ki_C is not None else Ki
        Kd_C_final = Kd_C if Kd_C is not None else Kd
        
        self.pid_A = PIDController(Kp=Kp_A_final, Ki=Ki_A_final, Kd=Kd_A_final, output_limit=output_limit)
        self.pid_B = PIDController(Kp=Kp_B_final, Ki=Ki_B_final, Kd=Kd_B_final, output_limit=output_limit)
        self.pid_C = PIDController(Kp=Kp_C_final, Ki=Ki_C_final, Kd=Kd_C_final, output_limit=output_limit)
        
        # Current state
        self.current_angles = np.array([neutral_angle, neutral_angle, neutral_angle])
        self.current_projections = np.array([0.0, 0.0, 0.0])
    
    def update(self, error_vector: np.ndarray, dt: float) -> np.ndarray:
        """
        Update controller and compute motor angles.
        
        Uses absolute control: calculates absolute angle position relative to neutral.
        The PID output is subtracted from neutral angle to provide negative feedback.
        This provides immediate response to errors and faster direction changes.
        
        Args:
            error_vector: 2D error vector [ex, ey] in meters
            dt: Time step in seconds
            
        Returns:
            Array of motor angles [angle_A, angle_B, angle_C] in degrees
        """
        # Project error onto motor axes
        projections = MotorGeometry.project_error(error_vector, self.motor_vectors)
        self.current_projections = projections
        
        # Update all three PID controllers simultaneously
        # Each motor uses its own independent PID controller with potentially different gains
        # PID output represents the change needed from current angle to eliminate error
        pid_output_A = self.pid_A.update(projections[0], dt)
        pid_output_B = self.pid_B.update(projections[1], dt)
        pid_output_C = self.pid_C.update(projections[2], dt)
        
        # Absolute control: angle relative to neutral, not incremental
        # This provides immediate response to errors and faster direction changes
        # PID output is subtracted from neutral to provide negative feedback
        angle_A = self.neutral_angle - pid_output_A
        angle_B = self.neutral_angle - pid_output_B
        angle_C = self.neutral_angle - pid_output_C
        
        # Clamp angles to valid servo range (0-65 degrees) for safety
        angles = np.array([angle_A, angle_B, angle_C])
        angles = np.clip(angles, 0.0, 65.0)
        
        self.current_angles = angles
        return angles
    
    def reset(self):
        """Reset all PID controllers and return angles to neutral."""
        self.pid_A.reset()
        self.pid_B.reset()
        self.pid_C.reset()
        # Reset current angles to neutral position
        self.current_angles = np.array([self.neutral_angle, self.neutral_angle, self.neutral_angle])
    
    def set_gains(self, Kp: float = None, Ki: float = None, Kd: float = None,
                  motor: str = 'all'):
        """
        Update PID gains for specified motor(s).
        
        Args:
            Kp, Ki, Kd: Gains to set (None to leave unchanged)
            motor: 'A', 'B', 'C', or 'all' (default: 'all')
        """
        if motor == 'A' or motor == 'all':
            if Kp is not None:
                self.pid_A.Kp = Kp
            if Ki is not None:
                self.pid_A.Ki = Ki
            if Kd is not None:
                self.pid_A.Kd = Kd
        
        if motor == 'B' or motor == 'all':
            if Kp is not None:
                self.pid_B.Kp = Kp
            if Ki is not None:
                self.pid_B.Ki = Ki
            if Kd is not None:
                self.pid_B.Kd = Kd
        
        if motor == 'C' or motor == 'all':
            if Kp is not None:
                self.pid_C.Kp = Kp
            if Ki is not None:
                self.pid_C.Ki = Ki
            if Kd is not None:
                self.pid_C.Kd = Kd


# ============================================================================
# Arduino Servo Controller
# ============================================================================

class ArduinoServoController:
    """
    Controls Stewart platform servos via Arduino serial interface.
    """
    
    def __init__(self, port=None, baud=115200):
        """
        Initialize Arduino servo controller.
        
        Args:
            port: Serial port (auto-detected if None, or specify manually like '/dev/tty.usbmodem14101')
            baud: Baud rate (default: 115200)
        """
        self.ser = None
        self.connected = False
        
        # Find and connect to Arduino
        if port is None:
            port = self.find_arduino_port()
        
        if port:
            self.connect(port, baud)
        else:
            print("[ARDUINO] No Arduino found - running in simulation mode")
            # List available ports to help user
            ports = serial.tools.list_ports.comports()
            if ports:
                print("[ARDUINO] Available serial ports:")
                for p in ports:
                    print(f"  - {p.device}: {p.description}")
                print("[ARDUINO] To connect manually, specify port: python directDrive.py --port /dev/tty.usbmodemXXX")
            self.connected = False
    
    def find_arduino_port(self):
        """Auto-detect Arduino port."""
        ports = serial.tools.list_ports.comports()
        for p in ports:
            # Look for common Arduino USB identifiers (more permissive)
            device_lower = p.device.lower()
            desc_lower = str(p.description).lower()
            if ('usbmodem' in device_lower or 'usbserial' in device_lower or 
                'arduino' in desc_lower or 'ch340' in desc_lower or 'ftdi' in desc_lower or
                'cp210' in desc_lower):
                print(f"[ARDUINO] Found Arduino on: {p.device}")
                return p.device
        # If no match found, try common ports (for manual connection)
        common_ports = ['/dev/tty.usbmodem', '/dev/tty.usbserial', '/dev/ttyUSB0', '/dev/ttyACM0']
        for port_pattern in common_ports:
            ports = serial.tools.list_ports.comports()
            for p in ports:
                if port_pattern.split('/')[-1] in p.device:
                    print(f"[ARDUINO] Found potential Arduino on: {p.device}")
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
        """
        Send servo angles to Arduino in degrees.
        Matches the exact format from arduino_controller.py
        
        Args:
            angles_deg: Array of 3 angles in degrees [angle_A, angle_B, angle_C]
            
        Returns:
            True if sent successfully, False otherwise
        """
        if self.ser is None:
            return False
        
        if not self.ser.is_open:
            # Try to reopen if closed
            try:
                self.ser.open()
            except:
                return False
        
        try:
            # Format: "angle0,angle1,angle2\n" - use int() to match arduino_controller.py exactly
            command = f"{int(angles_deg[0])},{int(angles_deg[1])},{int(angles_deg[2])}\n"
            bytes_written = self.ser.write(command.encode())
            
            if bytes_written == 0:
                return False
            
            # Non-blocking response check - don't wait, just check if data is available
            if self.ser.in_waiting:
                response = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if response.startswith('OK:'):
                    return True
                elif response.startswith('ERR:'):
                    print(f"[ARDUINO] Error: {response}")
                    return False
            # Assume success if no response yet (non-blocking, fast execution)
            return True
        except Exception as e:
            print(f"[ARDUINO] Send error: {e}")
            return False
    
    def close(self):
        """Close serial connection."""
        if self.ser and self.ser.is_open:
            # Reset to neutral before closing
            self.send_angles([15, 15, 15])  # Approximate neutral
            time.sleep(0.1)
            self.ser.close()
            print("[ARDUINO] Serial connection closed")


# ============================================================================
# Camera Manager
# ============================================================================

class CameraManager:
    """
    Manages camera capture and ball detection.
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
        """Get current ball position in meters."""
        return self.ball_position
    
    def get_camera_frame(self) -> Optional[np.ndarray]:
        """Get current camera frame."""
        return self.current_frame
    
    def close(self):
        """Release camera."""
        if self.cap is not None:
            self.cap.release()
            print("[CAMERA] Camera released")


# ============================================================================
# UI Manager
# ============================================================================

class UIManager:
    """
    Manages OpenCV UI overlay for direct-drive control.
    Shows motor vectors, error vector, projections, and motor angles.
    Includes interactive PID tuning control panel.
    """
    
    def __init__(self, camera_manager: CameraManager,
                 plate_radius_m: float = 0.15, frame_width: int = 1920, frame_height: int = 1440):
        """
        Initialize UI manager.
        
        Args:
            camera_manager: CameraManager instance
            plate_radius_m: Plate radius in meters (for pixel-to-platform conversion)
            frame_width: Camera frame width in pixels
            frame_height: Camera frame height in pixels
        """
        self.camera_manager = camera_manager
        
        # Get calibration from ball detector
        detector = camera_manager.detector
        
        # Use calibrated coordinate frame if available
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
        self.window_name = "Direct-Drive Stewart Platform Control"
        self.control_window = "PID Tuning"
        
        # Get motor vectors for visualization
        self.motor_vectors = MotorGeometry.get_motor_unit_vectors()
        
        # Control panel state (will be initialized by create_control_panel)
        self.selected_param = 0
        # Parameter names: Motor A, B, C each have Kp, Ki, Kd, plus CenterTol
        self.param_names = [
            'Kp_A', 'Ki_A', 'Kd_A',
            'Kp_B', 'Ki_B', 'Kd_B',
            'Kp_C', 'Ki_C', 'Kd_C',
            'CenterTol'
        ]
        self.editing_mode = False
        self.edit_buffer = ""
        self.reset_callback = None
        self.direct_controller = None  # Will be set by set_controller
        self.centered_tolerance = 0.010  # Default centered tolerance
    
    def pixel_to_platform(self, u: int, v: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to platform coordinates (meters).
        """
        if self.has_calibration:
            ball_px = np.array([u, v], dtype=np.float64)
            delta_px = ball_px - self.origin_px
            
            x_m = np.dot(delta_px, self.x_axis) * self.pixel_to_meter_ratio
            y_m = np.dot(delta_px, self.y_axis) * self.pixel_to_meter_ratio
        else:
            dx_px = u - self.plate_center_px[0]
            dy_px = self.plate_center_px[1] - v
            x_m = dx_px * self.pixel_to_meter_ratio
            y_m = dy_px * self.pixel_to_meter_ratio
        
        return (x_m, y_m)
    
    def platform_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """
        Convert platform coordinates (meters) to pixel coordinates.
        """
        if self.has_calibration:
            delta_px = (x_m / self.pixel_to_meter_ratio) * self.x_axis + \
                      (y_m / self.pixel_to_meter_ratio) * self.y_axis
            pixel_pos = self.origin_px + delta_px
            u = int(pixel_pos[0])
            v = int(pixel_pos[1])
        else:
            u = int(self.plate_center_px[0] + x_m / self.pixel_to_meter_ratio)
            v = int(self.plate_center_px[1] - y_m / self.pixel_to_meter_ratio)
        
        return (u, v)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to set target position."""
        if event == cv2.EVENT_LBUTTONDOWN:
            x_target, y_target = self.pixel_to_platform(x, y)
            # Store target in param (ControlState)
            if param is not None:
                param['x_target'] = x_target
                param['y_target'] = y_target
                print(f"[TARGET] New target set: ({x_target:.4f}, {y_target:.4f}) m")
    
    def set_controller(self, direct_controller: DirectDriveController):
        """Set reference to direct drive controller for parameter access."""
        self.direct_controller = direct_controller
    
    def set_reset_callback(self, callback):
        """Set callback function to reset PID controllers when parameters change."""
        self.reset_callback = callback
    
    def create_control_panel(self):
        """Create control panel window for parameter adjustment."""
        cv2.namedWindow(self.control_window)
        self.selected_param = 0
        self.editing_mode = False
        self.edit_buffer = ""
    
    def get_param_value(self, param_name: str) -> float:
        """Get current value of a parameter."""
        if self.direct_controller is None:
            return 0.0
        
        # Motor A parameters
        if param_name == 'Kp_A':
            return self.direct_controller.pid_A.Kp
        elif param_name == 'Ki_A':
            return self.direct_controller.pid_A.Ki
        elif param_name == 'Kd_A':
            return self.direct_controller.pid_A.Kd
        # Motor B parameters
        elif param_name == 'Kp_B':
            return self.direct_controller.pid_B.Kp
        elif param_name == 'Ki_B':
            return self.direct_controller.pid_B.Ki
        elif param_name == 'Kd_B':
            return self.direct_controller.pid_B.Kd
        # Motor C parameters
        elif param_name == 'Kp_C':
            return self.direct_controller.pid_C.Kp
        elif param_name == 'Ki_C':
            return self.direct_controller.pid_C.Ki
        elif param_name == 'Kd_C':
            return self.direct_controller.pid_C.Kd
        # Center tolerance
        elif param_name == 'CenterTol':
            return self.centered_tolerance
        return 0.0
    
    def set_param_value(self, param_name: str, value: float):
        """Set value of a parameter and reset PID controllers to clear accumulated errors."""
        value = max(0.0, value)  # Ensure non-negative
        needs_reset = False
        
        if self.direct_controller is None:
            return
        
        # Motor A parameters
        if param_name == 'Kp_A':
            value = min(value, 100.0)  # Max limit
            self.direct_controller.set_gains(Kp=value, motor='A')
            needs_reset = True
        elif param_name == 'Ki_A':
            value = min(value, 50.0)  # Max limit
            self.direct_controller.set_gains(Ki=value, motor='A')
            needs_reset = True
        elif param_name == 'Kd_A':
            value = min(value, 50.0)  # Max limit
            self.direct_controller.set_gains(Kd=value, motor='A')
            needs_reset = True
        # Motor B parameters
        elif param_name == 'Kp_B':
            value = min(value, 100.0)  # Max limit
            self.direct_controller.set_gains(Kp=value, motor='B')
            needs_reset = True
        elif param_name == 'Ki_B':
            value = min(value, 50.0)  # Max limit
            self.direct_controller.set_gains(Ki=value, motor='B')
            needs_reset = True
        elif param_name == 'Kd_B':
            value = min(value, 50.0)  # Max limit
            self.direct_controller.set_gains(Kd=value, motor='B')
            needs_reset = True
        # Motor C parameters
        elif param_name == 'Kp_C':
            value = min(value, 100.0)  # Max limit
            self.direct_controller.set_gains(Kp=value, motor='C')
            needs_reset = True
        elif param_name == 'Ki_C':
            value = min(value, 50.0)  # Max limit
            self.direct_controller.set_gains(Ki=value, motor='C')
            needs_reset = True
        elif param_name == 'Kd_C':
            value = min(value, 50.0)  # Max limit
            self.direct_controller.set_gains(Kd=value, motor='C')
            needs_reset = True
        # Center tolerance
        elif param_name == 'CenterTol':
            self.centered_tolerance = min(value, 0.05)  # Max 5cm
            # CenterTol doesn't need PID reset
        
        # Reset PID controllers if a control parameter was changed
        if needs_reset and self.reset_callback is not None:
            self.reset_callback()
            print(f"[RESET] PID integrals cleared after parameter change")
    
    def update_control_panel(self):
        """Update and display the control panel."""
        # Create blank image for control panel (taller to fit all motors)
        panel = np.zeros((650, 600, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark gray background
        
        # Title
        cv2.putText(panel, "PID Control Panel - All Motors", (120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        y_pos = 55
        instructions = [
            "W/S or UP/DOWN: Select parameter",
            "ENTER: Edit value | ESC: Cancel",
            "All motors controlled simultaneously"
        ]
        for instruction in instructions:
            cv2.putText(panel, instruction, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += 18
        
        # Draw separator line
        cv2.line(panel, (10, y_pos), (590, y_pos), (100, 100, 100), 1)
        y_pos += 15
        
        # Motor colors for visual distinction
        motor_colors = {
            'A': (0, 0, 255),    # Red
            'B': (0, 255, 0),    # Green
            'C': (255, 0, 0)     # Blue
        }
        
        # Display parameters grouped by motor
        current_motor = None
        for i, param_name in enumerate(self.param_names):
            # Extract motor from parameter name
            if param_name.startswith('Kp_') or param_name.startswith('Ki_') or param_name.startswith('Kd_'):
                motor = param_name.split('_')[1]
                if motor != current_motor:
                    # New motor section - add header
                    if current_motor is not None:
                        y_pos += 5  # Extra space between motors
                    current_motor = motor
                    motor_color = motor_colors.get(motor, (255, 255, 255))
                    cv2.putText(panel, f"Motor {motor}:", (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, motor_color, 2)
                    y_pos += 25
            
            is_selected = (i == self.selected_param)
            value = self.get_param_value(param_name)
            
            # Background highlight for selected parameter
            if is_selected:
                cv2.rectangle(panel, (5, y_pos - 18), (595, y_pos + 3), (80, 80, 80), -1)
            
            # Parameter name
            if param_name == 'CenterTol':
                color = (0, 255, 255) if is_selected else (255, 255, 255)
            else:
                motor = param_name.split('_')[1] if '_' in param_name else None
                color = motor_colors.get(motor, (255, 255, 255)) if motor else (255, 255, 255)
                if is_selected:
                    color = (0, 255, 255)  # Cyan when selected
            
            param_display = param_name.replace('_', ' ')
            cv2.putText(panel, f"{param_display}:", (30, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if is_selected else 1)
            
            # Parameter value (show edit buffer if editing this parameter)
            if is_selected and self.editing_mode:
                value_text = self.edit_buffer + "_"
                value_color = (0, 255, 0)  # Green when editing
            else:
                value_text = f"{value:.3f}"
                value_color = (255, 255, 255)
            
            cv2.putText(panel, value_text, (300, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, value_color, 2 if is_selected else 1)
            
            y_pos += 22
        
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
    
    def draw_overlay(self, frame: Optional[np.ndarray], 
                     x_target: float, y_target: float,
                     ball_pos: Optional[Tuple[float, float]],
                     error_vector: np.ndarray,
                     projections: np.ndarray,
                     motor_angles: np.ndarray,
                     centered: bool,
                     direct_controller: Optional[DirectDriveController] = None) -> Optional[np.ndarray]:
        """
        Draw control overlay on camera frame.
        
        Args:
            frame: Input BGR frame
            x_target: Target X position in meters
            y_target: Target Y position in meters
            ball_pos: Ball position (x, y) in meters or None
            error_vector: Error vector [ex, ey] in meters
            projections: Projections [proj_A, proj_B, proj_C]
            motor_angles: Motor angles [angle_A, angle_B, angle_C] in degrees
            centered: Whether ball is centered
            
        Returns:
            Frame with overlay drawn
        """
        if frame is None:
            return None
        
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        center_x, center_y = self.plate_center_px
        
        # Draw platform center crosshair
        cv2.line(overlay, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 255), 2)
        cv2.line(overlay, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 255), 2)
        cv2.circle(overlay, (center_x, center_y), 5, (0, 255, 255), -1)
        cv2.putText(overlay, "Origin", (center_x + 10, center_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw coordinate axes if calibrated
        if self.has_calibration:
            axis_length = 80
            x_end = (int(center_x + self.x_axis[0] * axis_length),
                    int(center_y + self.x_axis[1] * axis_length))
            cv2.arrowedLine(overlay, (center_x, center_y), x_end, (0, 0, 255), 2, tipLength=0.3)
            cv2.putText(overlay, "X", (x_end[0] + 5, x_end[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            y_end = (int(center_x + self.y_axis[0] * axis_length),
                    int(center_y + self.y_axis[1] * axis_length))
            cv2.arrowedLine(overlay, (center_x, center_y), y_end, (0, 255, 0), 2, tipLength=0.3)
            cv2.putText(overlay, "Y", (y_end[0] + 5, y_end[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw motor direction vectors (scaled for visualization)
        motor_scale = 100  # pixels
        motor_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: Red, Green, Blue
        motor_labels = ['A', 'B', 'C']
        
        for i, (motor_vec, color, label) in enumerate(zip(self.motor_vectors, motor_colors, motor_labels)):
            # Convert motor vector to pixel coordinates
            if self.has_calibration:
                vec_px = motor_vec[0] * self.x_axis * motor_scale / self.pixel_to_meter_ratio + \
                        motor_vec[1] * self.y_axis * motor_scale / self.pixel_to_meter_ratio
            else:
                vec_px = np.array([motor_vec[0] * motor_scale / self.pixel_to_meter_ratio,
                                  -motor_vec[1] * motor_scale / self.pixel_to_meter_ratio])
            
            motor_end = (int(center_x + vec_px[0]), int(center_y + vec_px[1]))
            cv2.arrowedLine(overlay, (center_x, center_y), motor_end, color, 2, tipLength=0.2)
            cv2.putText(overlay, f"M{label}", (motor_end[0] + 5, motor_end[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw target position
        target_x_px, target_y_px = self.platform_to_pixel(x_target, y_target)
        if 0 <= target_x_px < w and 0 <= target_y_px < h:
            cv2.drawMarker(overlay, (target_x_px, target_y_px), (0, 255, 0), 
                          cv2.MARKER_CROSS, 20, 2)
            cv2.putText(overlay, "Target", (target_x_px + 5, target_y_px - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw ball position
        if ball_pos is not None:
            x, y = ball_pos
            ball_x_px, ball_y_px = self.platform_to_pixel(x, y)
            if 0 <= ball_x_px < w and 0 <= ball_y_px < h:
                ball_color = (0, 255, 0) if centered else (255, 0, 255)
                cv2.circle(overlay, (ball_x_px, ball_y_px), 10, ball_color, 2)
                status_text = "CENTERED" if centered else f"({x:.3f}, {y:.3f})m"
                cv2.putText(overlay, f"Ball: {status_text}", 
                           (ball_x_px + 15, ball_y_px), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, ball_color, 2)
        
        # Draw error vector (arrow from ball to target)
        if ball_pos is not None:
            error_scale = 50  # pixels per meter
            if self.has_calibration:
                error_px = error_vector[0] * self.x_axis * error_scale / self.pixel_to_meter_ratio + \
                          error_vector[1] * self.y_axis * error_scale / self.pixel_to_meter_ratio
            else:
                error_px = np.array([error_vector[0] * error_scale / self.pixel_to_meter_ratio,
                                    -error_vector[1] * error_scale / self.pixel_to_meter_ratio])
            
            error_start = (ball_x_px, ball_y_px) if ball_pos is not None else (center_x, center_y)
            error_end = (int(error_start[0] + error_px[0]), int(error_start[1] + error_px[1]))
            cv2.arrowedLine(overlay, error_start, error_end, (0, 165, 255), 3, tipLength=0.3)
            cv2.putText(overlay, "Error", (error_end[0] + 5, error_end[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        # Draw status text
        y_offset = 30
        cv2.putText(overlay, f"Target: ({x_target:.3f}, {y_target:.3f}) m",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        
        if ball_pos is not None:
            x, y = ball_pos
            ex, ey = error_vector
            error_mag = np.sqrt(ex**2 + ey**2)
            error_color = (0, 255, 0) if centered else (255, 255, 0)
            status_str = " [CENTERED]" if centered else ""
            cv2.putText(overlay, f"Error: {error_mag:.4f}m ({ex:.3f}, {ey:.3f}){status_str}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, error_color, 2)
            y_offset += 25
        
        # Draw projections
        y_offset += 10
        cv2.putText(overlay, "Projections:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        proj_labels = ['A', 'B', 'C']
        for i, (proj, label) in enumerate(zip(projections, proj_labels)):
            cv2.putText(overlay, f"  M{label}: {proj:.4f} m",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, motor_colors[i], 1)
            y_offset += 18
        
        # Draw motor angles
        y_offset += 10
        cv2.putText(overlay, "Motor Angles:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        for i, (angle, label) in enumerate(zip(motor_angles, proj_labels)):
            cv2.putText(overlay, f"  M{label}: {angle:.1f}°",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, motor_colors[i], 1)
            y_offset += 18
        
        # Draw PID gains for each motor (if controller is available)
        if direct_controller is not None:
            y_offset += 10
            cv2.putText(overlay, "PID Gains:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
            # Motor A (Red)
            cv2.putText(overlay, f"Motor A: Kp={direct_controller.pid_A.Kp:.2f} Ki={direct_controller.pid_A.Ki:.2f} Kd={direct_controller.pid_A.Kd:.2f}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            y_offset += 16
            
            # Motor B (Green)
            cv2.putText(overlay, f"Motor B: Kp={direct_controller.pid_B.Kp:.2f} Ki={direct_controller.pid_B.Ki:.2f} Kd={direct_controller.pid_B.Kd:.2f}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            y_offset += 16
            
            # Motor C (Blue)
            cv2.putText(overlay, f"Motor C: Kp={direct_controller.pid_C.Kp:.2f} Ki={direct_controller.pid_C.Ki:.2f} Kd={direct_controller.pid_C.Kd:.2f}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        
        # Instructions
        y_offset = h - 40
        cv2.putText(overlay, "Click to set target | 'q' quit | 'r' reset | W/S to tune PID",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return overlay


# ============================================================================
# Data Logger
# ============================================================================

class DataLogger:
    """
    Handles CSV logging of flight data.
    """
    def __init__(self, directory="logs"):
        # Create logs directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Generate filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(directory, f"direct_drive_{timestamp}.csv")
        
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        
        # Write Header
        self.writer.writerow([
            "Time_s",           # Elapsed time
            "Target_X_m",       # Desired X
            "Target_Y_m",       # Desired Y
            "Current_X_m",      # Actual X
            "Current_Y_m",      # Actual Y
            "Error_X_m",        # Displacement X
            "Error_Y_m",        # Displacement Y
            "Error_Dist_m",     # Total Error Magnitude
            "Proj_A_m",         # Projection on Motor A
            "Proj_B_m",         # Projection on Motor B
            "Proj_C_m",         # Projection on Motor C
            "PID_Out_A",        # PID output Motor A
            "PID_Out_B",        # PID output Motor B
            "PID_Out_C",        # PID output Motor C
            "Angle_A_deg",      # Motor A angle
            "Angle_B_deg",      # Motor B angle
            "Angle_C_deg",      # Motor C angle
            "Centered"          # Boolean flag
        ])
        print(f"[LOGGER] Recording data to: {self.filename}")

    def log_step(self, time_s, target_x, target_y, cur_x, cur_y, 
                 err_x, err_y, err_dist, projections, pid_outputs, motor_angles, centered):
        """
        Log one control step.
        
        Args:
            time_s: Elapsed time in seconds
            target_x: Target X position
            target_y: Target Y position
            cur_x: Current X position
            cur_y: Current Y position
            err_x: Error X
            err_y: Error Y
            err_dist: Error magnitude
            projections: Array [proj_A, proj_B, proj_C]
            pid_outputs: Array [pid_A, pid_B, pid_C]
            motor_angles: Array [angle_A, angle_B, angle_C]
            centered: Whether ball is centered
        """
        self.writer.writerow([
            f"{time_s:.4f}",
            f"{target_x:.4f}",
            f"{target_y:.4f}",
            f"{cur_x:.4f}",
            f"{cur_y:.4f}",
            f"{err_x:.4f}",
            f"{err_y:.4f}",
            f"{err_dist:.4f}",
            f"{projections[0]:.4f}",
            f"{projections[1]:.4f}",
            f"{projections[2]:.4f}",
            f"{pid_outputs[0]:.4f}",
            f"{pid_outputs[1]:.4f}",
            f"{pid_outputs[2]:.4f}",
            f"{motor_angles[0]:.2f}",
            f"{motor_angles[1]:.2f}",
            f"{motor_angles[2]:.2f}",
            1 if centered else 0
        ])

    def close(self):
        if self.file:
            self.file.close()
            print(f"[LOGGER] Log saved to {self.filename}")


# ============================================================================
# Control Loop
# ============================================================================

class ControlLoop:
    """
    Main control loop that coordinates all components.
    """
    
    def __init__(self, camera_manager: CameraManager,
                 direct_controller: DirectDriveController,
                 ui_manager: UIManager,
                 servo_controller: ArduinoServoController,
                 target_dt: float = 0.0):
        """
        Initialize control loop.
        
        Args:
            camera_manager: CameraManager instance
            direct_controller: DirectDriveController instance
            ui_manager: UIManager instance
            servo_controller: ArduinoServoController instance
            target_dt: Target loop time in seconds (0.0 = max speed)
        """
        self.camera_manager = camera_manager
        self.direct_controller = direct_controller
        self.ui_manager = ui_manager
        self.servo_controller = servo_controller
        self.target_dt = target_dt
        
        # Initialize Data Logger
        self.logger = DataLogger()
        self.start_time = None
        
        # Control state
        self.x_target = 0.0
        self.y_target = 0.0
        self.running = False
        self.emergency_stop = False
        self.last_update_time = None
        
        # Centered tolerance (will be synced with UI manager)
        self.centered_tolerance = 0.010  # meters (1cm margin)
        
        # Rate-limited target position
        self.x_target_filtered = 0.0
        self.y_target_filtered = 0.0
        self.max_target_rate = 5.0  # m/s
        
        # Store PID outputs for logging
        self.last_pid_outputs = np.array([0.0, 0.0, 0.0])
        
        # Store last motor angles for when ball is not detected
        self._last_motor_angles = None
        
        # Performance optimization counters
        self._loop_counter = 0
        self._panel_update_counter = 0
        self._ui_update_counter = 0
        self._last_angles_int = None
        self._fps_start_time = None
        self._timings = {'camera': [], 'control': [], 'arduino': [], 'ui': [], 'total': []}
    
    def run(self):
        """Run main control loop."""
        print("="*70)
        print("Direct-Drive Stewart Platform Controller")
        print("="*70)
        print("Controls:")
        print("  - Click on camera feed to set target position")
        print("  - Use W/S or UP/DOWN arrows to select parameter in control panel")
        print("  - Press ENTER to edit selected parameter")
        print("  - Type value and press ENTER to confirm")
        print("  - Press 'q' to quit, 'r' to reset PID integrals")
        print("  - PID automatically resets when Kp/Ki/Kd changed")
        print("="*70)
        
        # Initialize UI
        cv2.namedWindow(self.ui_manager.window_name)
        self.ui_manager.create_control_panel()
        self.ui_manager.set_controller(self.direct_controller)
        
        # Set up callback for PID reset when parameters change
        def reset_pids():
            self.direct_controller.reset()
        self.ui_manager.set_reset_callback(reset_pids)
        
        # Mouse callback with target storage
        target_state = {'x_target': self.x_target, 'y_target': self.y_target}
        cv2.setMouseCallback(self.ui_manager.window_name, 
                            lambda e, x, y, f, p: self.ui_manager.mouse_callback(e, x, y, f, target_state))
        
        self.running = True
        self.start_time = time.time()
        self.last_update_time = time.time()
        
        # Initialize filtered setpoint
        self.x_target_filtered = self.x_target
        self.y_target_filtered = self.y_target
        
        # Initialize with neutral angles and send to Arduino
        neutral_angles = np.array([self.direct_controller.neutral_angle,
                                  self.direct_controller.neutral_angle,
                                  self.direct_controller.neutral_angle])
        self.servo_controller.send_angles(neutral_angles)
        self._last_motor_angles = neutral_angles.copy()
        print(f"[INIT] Sent initial neutral angles: [{neutral_angles[0]:.1f}°, {neutral_angles[1]:.1f}°, {neutral_angles[2]:.1f}°]")
        
        # Control loop
        while self.running and not self.emergency_stop:
            loop_start = time.time()
            self._loop_counter += 1
            
            # Initialize performance monitoring
            if self._fps_start_time is None:
                self._fps_start_time = time.time()
            
            # Update target from mouse callback
            self.x_target = target_state['x_target']
            self.y_target = target_state['y_target']
            
            # Update camera and ball detection (time this)
            cam_start = time.time()
            self.camera_manager.update()
            self._timings['camera'].append(time.time() - cam_start)
            
            # Update control panel display (every 3rd frame for performance)
            self._panel_update_counter += 1
            if self._panel_update_counter % 3 == 0:
                self.ui_manager.update_control_panel()
            
            # Sync centered tolerance from UI manager
            self.centered_tolerance = self.ui_manager.centered_tolerance
            
            # Get ball position
            ball_pos = self.camera_manager.get_ball_position()
            
            # Calculate dt
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time
            
            # Clamp dt to reasonable range
            dt = np.clip(dt, 0.001, 0.1)
            
            # Apply rate limiting to target setpoint
            max_delta = self.max_target_rate * dt
            
            dx_target = self.x_target - self.x_target_filtered
            if abs(dx_target) > max_delta:
                dx_target = np.sign(dx_target) * max_delta
            self.x_target_filtered += dx_target
            
            dy_target = self.y_target - self.y_target_filtered
            if abs(dy_target) > max_delta:
                dy_target = np.sign(dy_target) * max_delta
            self.y_target_filtered += dy_target
            
            # Control update
            motor_angles = None
            
            if ball_pos is not None:
                x, y = ball_pos
                
                # Compute error vector
                ex = self.x_target_filtered - x
                ey = self.y_target_filtered - y
                error_vector = np.array([ex, ey])
                
                # Check if ball is centered
                error_magnitude = np.sqrt(ex**2 + ey**2)
                is_centered = error_magnitude < self.centered_tolerance
                
                # Store actual error for display (use view instead of copy for performance)
                display_error_vector = error_vector
                
                # If centered, use zero error to maintain PID state without applying control
                # if is_centered:
                #     error_vector = np.array([0.0, 0.0])
                
                # Update controller (time this)
                ctrl_start = time.time()
                motor_angles = self.direct_controller.update(error_vector, dt)
                self._timings['control'].append(time.time() - ctrl_start)
                
                # Extract PID outputs (angle deltas from neutral)
                self.last_pid_outputs = motor_angles - self.direct_controller.neutral_angle
                
                # If centered, ensure angles are exactly at neutral
                # if is_centered:
                #     motor_angles = np.array([self.direct_controller.neutral_angle,
                #                            self.direct_controller.neutral_angle,
                #                            self.direct_controller.neutral_angle])
                #     self.last_pid_outputs = np.array([0.0, 0.0, 0.0])
                
                # Log data (every 2nd frame to reduce I/O overhead)
                if self._loop_counter % 2 == 0:
                    elapsed_time = time.time() - self.start_time
                    self.logger.log_step(
                        time_s=elapsed_time,
                        target_x=self.x_target_filtered,
                        target_y=self.y_target_filtered,
                        cur_x=x,
                        cur_y=y,
                        err_x=ex,
                        err_y=ey,
                        err_dist=error_magnitude,
                        projections=self.direct_controller.current_projections,
                        pid_outputs=self.last_pid_outputs,
                        motor_angles=motor_angles,
                        centered=is_centered
                    )
            else:
                # Ball not detected - maintain last angles or use neutral
                if not hasattr(self, '_last_motor_angles') or self._last_motor_angles is None:
                    motor_angles = np.array([self.direct_controller.neutral_angle,
                                           self.direct_controller.neutral_angle,
                                           self.direct_controller.neutral_angle])
                else:
                    motor_angles = self._last_motor_angles.copy()
            
            # Send angles to hardware (only if changed to reduce communication overhead)
            if motor_angles is not None:
                # Convert to integers (matching arduino_controller.py format)
                angles_int = [int(motor_angles[0]), int(motor_angles[1]), int(motor_angles[2])]
                
                # Only send if angles changed (optimization)
                if angles_int != self._last_angles_int:
                    arduino_start = time.time()
                    sent = self.servo_controller.send_angles(angles_int)
                    self._timings['arduino'].append(time.time() - arduino_start)
                    self._last_angles_int = angles_int
                else:
                    sent = True  # Assume success if no change needed
                
                # Store for next iteration (only copy if needed)
                self._last_motor_angles = motor_angles
                
                # Warn if commands aren't being sent (only once to avoid spam)
                if not sent and not hasattr(self, '_warned_no_arduino'):
                    if self.servo_controller.ser is None:
                        print("[WARNING] Arduino not connected - commands are NOT being sent to hardware!")
                        print("[WARNING] Check Arduino connection or specify port with --port argument")
                        self._warned_no_arduino = True
                
                # Debug output every 30 iterations (about once per second at 30fps)
                if not hasattr(self, '_debug_counter'):
                    self._debug_counter = 0
                self._debug_counter += 1
                if self._debug_counter % 30 == 0:
                    # Show rounded angles that will actually be sent
                    if ball_pos is not None:
                        arduino_status = "CONNECTED" if (self.servo_controller.connected and self.servo_controller.ser and self.servo_controller.ser.is_open) else "SIMULATION"
                        ser_status = "OPEN" if (self.servo_controller.ser and self.servo_controller.ser.is_open) else "CLOSED/NONE"
                        print(f"[SERVO] Arduino: {arduino_status} (Serial: {ser_status}) | "
                              f"Error: {error_magnitude:.4f}m | "
                              f"Projections: [{self.direct_controller.current_projections[0]:.4f}, "
                              f"{self.direct_controller.current_projections[1]:.4f}, "
                              f"{self.direct_controller.current_projections[2]:.4f}]m | "
                              f"PID outputs: [{self.last_pid_outputs[0]:.2f}, {self.last_pid_outputs[1]:.2f}, {self.last_pid_outputs[2]:.2f}]° | "
                              f"Angles (sent): {angles_int}° | "
                              f"Sent: {sent}")
                    else:
                        arduino_status = "CONNECTED" if (self.servo_controller.connected and self.servo_controller.ser and self.servo_controller.ser.is_open) else "SIMULATION"
                        ser_status = "OPEN" if (self.servo_controller.ser and self.servo_controller.ser.is_open) else "CLOSED/NONE"
                        print(f"[SERVO] Arduino: {arduino_status} (Serial: {ser_status}) | No ball detected, maintaining angles: {angles_int}° | Sent: {sent}")
                
            # Update display (every 2nd frame for performance)
            self._ui_update_counter += 1
            if self._ui_update_counter % 2 == 0:
                frame = self.camera_manager.get_camera_frame()
                if frame is not None:
                    ui_start = time.time()
                    # Prepare display data
                    if ball_pos is not None:
                        display_error = display_error_vector
                        is_centered_display = is_centered
                    else:
                        display_error = np.array([0.0, 0.0])
                        is_centered_display = False
                    
                    overlay = self.ui_manager.draw_overlay(
                        frame=frame,
                        x_target=self.x_target_filtered,
                        y_target=self.y_target_filtered,
                        ball_pos=ball_pos,
                        error_vector=display_error,
                        projections=self.direct_controller.current_projections if ball_pos is not None else np.array([0.0, 0.0, 0.0]),
                        motor_angles=motor_angles if motor_angles is not None else np.array([self.direct_controller.neutral_angle] * 3),
                        centered=is_centered_display,
                        direct_controller=self.direct_controller
                    )
                    if overlay is not None:
                        cv2.imshow(self.ui_manager.window_name, overlay)
                    self._timings['ui'].append(time.time() - ui_start)
            
            # Handle keyboard input (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            
            # Try to handle key in control panel first
            if key != 255:  # 255 means no key pressed
                handled = self.ui_manager.handle_control_key(key)
                if not handled:
                    # Handle global keys if not handled by control panel
                    if key == ord('q'):
                        print("[STOP] Emergency stop requested")
                        self.emergency_stop = True
                        break
                    elif key == ord('r'):
                        print("[RESET] Resetting PID integrals")
                        self.direct_controller.reset()
            
            # Performance monitoring and reporting
            elapsed = time.time() - loop_start
            self._timings['total'].append(elapsed)
            
            # Print performance stats every 100 loops
            if self._loop_counter % 100 == 0:
                elapsed_total = time.time() - self._fps_start_time
                fps = 100 / elapsed_total if elapsed_total > 0 else 0
                
                # Calculate average timings
                avg_camera = np.mean(self._timings['camera']) * 1000 if self._timings['camera'] else 0
                avg_control = np.mean(self._timings['control']) * 1000 if self._timings['control'] else 0
                avg_arduino = np.mean(self._timings['arduino']) * 1000 if self._timings['arduino'] else 0
                avg_ui = np.mean(self._timings['ui']) * 1000 if self._timings['ui'] else 0
                avg_total = np.mean(self._timings['total']) * 1000 if self._timings['total'] else 0
                
                print(f"[PERF] Loop: {fps:.1f} Hz | "
                      f"Camera: {avg_camera:.1f}ms | "
                      f"Control: {avg_control:.1f}ms | "
                      f"Arduino: {avg_arduino:.1f}ms | "
                      f"UI: {avg_ui:.1f}ms | "
                      f"Total: {avg_total:.1f}ms")
                
                # Reset counters
                self._fps_start_time = time.time()
                self._timings = {'camera': [], 'control': [], 'arduino': [], 'ui': [], 'total': []}
        
        # Cleanup: set platform to neutral
        print("[CLEANUP] Setting platform to neutral")
        neutral_angles = np.array([self.direct_controller.neutral_angle,
                                  self.direct_controller.neutral_angle,
                                  self.direct_controller.neutral_angle])
        self.servo_controller.send_angles(neutral_angles)
        self.camera_manager.close()
        self.logger.close()
        cv2.destroyAllWindows()
        self.running = False


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Direct-Drive Stewart Platform Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python directDrive.py                          # Run without camera calibration
  python directDrive.py cal/camera_calib.npz     # Run with camera calibration
  python directDrive.py --dt 0.1                # Run at fixed 10Hz
        """
    )
    parser.add_argument(
        'calib_file',
        nargs='?',
        default=None,
        help='Path to camera calibration file (.npz). If not provided, camera calibration is disabled.'
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.0,
        help='Force minimum sampling time (s) to simulate discrete/slow plant'
    )
    parser.add_argument(
        '--Kp',
        type=float,
        default=20.0,
        help='Proportional gain (default: 10.0, tuned for meter inputs)'
    )
    parser.add_argument(
        '--Ki',
        type=float,
        default=12.0,
        help='Integral gain (default: 5.0, tuned for meter inputs)'
    )
    parser.add_argument(
        '--Kd',
        type=float,
        default=15.0,
        help='Derivative gain (default: 8.0, tuned for meter inputs)'
    )
    parser.add_argument(
        '--port',
        type=str,
        default=None,
        help='Arduino serial port (e.g., /dev/tty.usbmodem14101). Auto-detected if not specified.'
    )
    args = parser.parse_args()
    
    print("="*70)
    print("Direct-Drive Stewart Platform Controller")
    print("="*70)
    
    if args.calib_file:
        print(f"[CONFIG] Camera calibration file: {args.calib_file}")
    else:
        print("[CONFIG] Camera calibration: DISABLED")
    
    if args.dt > 0:
        print(f"[CONFIG] Forced Sampling Time: {args.dt}s ({1/args.dt:.1f} Hz)")
    
    print(f"[CONFIG] PID Gains: Kp={args.Kp:.3f}, Ki={args.Ki:.3f}, Kd={args.Kd:.3f}")
    print("="*70)
    
    # Initialize components
    print("[INIT] Connecting to Arduino...")
    servo_controller = ArduinoServoController(port=args.port)
    
    print("[INIT] Initializing camera and ball detector...")
    camera_manager = CameraManager(camera_id=0, calib_file=args.calib_file)
    
    print("[INIT] Initializing direct-drive controller...")
    direct_controller = DirectDriveController(
        Kp=args.Kp,
        Ki=args.Ki,
        Kd=args.Kd,
        output_limit=20.0,
        neutral_angle=15.5
    )
    
    print("[INIT] Initializing UI manager...")
    ui_manager = UIManager(
        camera_manager=camera_manager,
        plate_radius_m=0.15,
        frame_width=1920,
        frame_height=1440
    )
    
    # Create and run control loop
    control_loop = ControlLoop(
        camera_manager=camera_manager,
        direct_controller=direct_controller,
        ui_manager=ui_manager,
        servo_controller=servo_controller,
        target_dt=args.dt
    )
    
    try:
        control_loop.run()
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user")
        control_loop.emergency_stop = True
        servo_controller.send_angles([15, 15, 15])
        camera_manager.close()
        servo_controller.close()
    except Exception as e:
        print(f"[ERROR] Control loop failed: {e}")
        import traceback
        traceback.print_exc()
        servo_controller.send_angles([15, 15, 15])
        camera_manager.close()
        servo_controller.close()


if __name__ == "__main__":
    main()

