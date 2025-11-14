#!/usr/bin/env python3
"""
3D Stewart Platform PID Controller
Real-time 2D PID control for ball balancing on a 3-motor Stewart platform.

This controller:
- Reads ball position (x, y) in meters from ball detection
- Uses independent PID controllers for x and y axes
- Maps PID outputs to a desired platform normal vector
- Provides live tuning via OpenCV trackbars
- Allows target selection via mouse clicks on camera feed
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple


# ============================================================================
# PID Controller Class
# ============================================================================

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


# ============================================================================
# Control State Management
# ============================================================================

class ControlState:
    """Manages control state: gains, target position, and runtime parameters."""
    
    def __init__(self):
        # Target position (meters, platform coordinates)
        self.x_target = 0.0
        self.y_target = 0.0
        
        # PID gains for x and y axes
        self.Kp_x = 10.0
        self.Ki_x = 0.5
        self.Kd_x = 2.0
        
        self.Kp_y = 10.0
        self.Ki_y = 0.5
        self.Kd_y = 2.0
        
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


# ============================================================================
# Ball Tracker (Hardware Interface Stub)
# ============================================================================

class BallTracker:
    """
    Wrapper around ball detection hardware interface.
    
    This is a STUB - replace get_ball_position() with your actual ball detection code.
    """
    
    def __init__(self):
        """Initialize ball tracker."""
        # STUB: In real implementation, initialize camera/ball detector here
        self.detector = None  # Replace with your BallDetector instance
        
    def get_ball_position(self) -> Optional[Tuple[float, float]]:
        """
        Get current ball position in platform coordinates.
        
        Returns:
            (x, y) tuple in meters, or None if ball not detected.
            Coordinate system: (0, 0) at plate center, x right, y up.
            
        STUB IMPLEMENTATION:
        Replace this with your actual ball detection code, e.g.:
            from ball_detection import BallDetector
            found, center, radius, (x_m, y_m) = self.detector.detect_ball(frame)
            if found:
                return (x_m, y_m)
            return None
        """
        # STUB: Return simulated position for testing
        # In real code, call your ball detection here
        return (0.01, -0.02)  # Example: ball slightly right and down
    
    def get_camera_frame(self) -> Optional[np.ndarray]:
        """
        Get current camera frame for display.
        
        Returns:
            BGR image array, or None if frame not available.
            
        STUB IMPLEMENTATION:
        Replace with your actual camera capture code.
        """
        # STUB: Return None or a dummy frame
        # In real code: ret, frame = cap.read(); return frame if ret else None
        return None


# ============================================================================
# Normal Controller (PID Output → Plane Normal)
# ============================================================================

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
# Hardware Interface Stub
# ============================================================================

def set_desired_normal(nx: float, ny: float, nz: float):
    """
    Set desired platform normal vector (hardware interface).
    
    This is a STUB - replace with your actual motor control code.
    
    Args:
        nx: x-component of unit normal vector
        ny: y-component of unit normal vector
        nz: z-component of unit normal vector (should be positive)
        
    STUB IMPLEMENTATION:
    In your real code, this should:
    1. Convert normal vector to platform orientation (e.g., roll/pitch angles)
    2. Use inverse kinematics to compute motor positions
    3. Send commands to motors
    
    Example:
        # Convert normal to roll/pitch
        roll = np.arcsin(-nx)
        pitch = np.arcsin(ny / np.cos(roll))
        
        # Use inverse kinematics (your code)
        motor_angles = inverse_kinematics(roll, pitch, 0.0)
        
        # Send to motors
        send_motor_commands(motor_angles)
    """
    # STUB: Just log the desired normal
    print(f"[HARDWARE] Desired normal: n=({nx:.4f}, {ny:.4f}, {nz:.4f})")
    # TODO: Replace with actual motor control code


# ============================================================================
# UI Manager (OpenCV Windows, Trackbars, Mouse Callbacks)
# ============================================================================

class UIManager:
    """
    Manages OpenCV UI: camera display, PID tuning trackbars, mouse callbacks.
    """
    
    def __init__(self, state: ControlState, ball_tracker: BallTracker,
                 plate_radius_m: float = 0.15, frame_width: int = 640, frame_height: int = 480):
        """
        Initialize UI manager.
        
        Args:
            state: ControlState instance
            ball_tracker: BallTracker instance
            plate_radius_m: Plate radius in meters (for pixel-to-platform conversion)
            frame_width: Camera frame width in pixels
            frame_height: Camera frame height in pixels
        """
        self.state = state
        self.ball_tracker = ball_tracker
        
        # Calibration parameters (assumed - should be calibrated)
        self.plate_radius_m = plate_radius_m
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.plate_center_px = (frame_width // 2, frame_height // 2)
        
        # Estimate plate radius in pixels (assume plate fills ~80% of frame)
        self.plate_radius_px = min(frame_width, frame_height) * 0.4
        
        # OpenCV windows
        self.window_name = "Stewart Platform Control"
        self.control_window = "PID Tuning"
        
        # Note: Display frame management removed - direct display in control loop
        
    def pixel_to_platform(self, u: int, v: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to platform coordinates (meters).
        
        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate
            
        Returns:
            (x, y) in meters, platform coordinates
            
        Assumptions:
        - Camera is mounted directly above platform center
        - Linear mapping: pixel distance from center → meters
        - Coordinate system: (0,0) at image center, x right, y up
        """
        # Convert to coordinates relative to image center
        dx_px = u - self.plate_center_px[0]
        dy_px = self.plate_center_px[1] - v  # Invert y (image y increases downward)
        
        # Scale to meters (linear mapping)
        scale = self.plate_radius_m / self.plate_radius_px
        x_m = dx_px * scale
        y_m = dy_px * scale
        
        return (x_m, y_m)
    
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
        
        # X-axis PID gains (trackbars use integers, scale appropriately)
        cv2.createTrackbar("Kp_x", self.control_window, 
                          max(0, min(500, int(self.state.Kp_x * 10))), 500, self.on_trackbar)
        cv2.createTrackbar("Ki_x", self.control_window, 
                          max(0, min(200, int(self.state.Ki_x * 100))), 200, self.on_trackbar)
        cv2.createTrackbar("Kd_x", self.control_window, 
                          max(0, min(100, int(self.state.Kd_x * 10))), 100, self.on_trackbar)
        
        # Y-axis PID gains
        cv2.createTrackbar("Kp_y", self.control_window, 
                          max(0, min(500, int(self.state.Kp_y * 10))), 500, self.on_trackbar)
        cv2.createTrackbar("Ki_y", self.control_window, 
                          max(0, min(200, int(self.state.Ki_y * 100))), 200, self.on_trackbar)
        cv2.createTrackbar("Kd_y", self.control_window, 
                          max(0, min(100, int(self.state.Kd_y * 10))), 100, self.on_trackbar)
        
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
        self.state.Kp_x = cv2.getTrackbarPos("Kp_x", self.control_window) / 10.0
        self.state.Ki_x = cv2.getTrackbarPos("Ki_x", self.control_window) / 100.0
        self.state.Kd_x = cv2.getTrackbarPos("Kd_x", self.control_window) / 10.0
        
        self.state.Kp_y = cv2.getTrackbarPos("Kp_y", self.control_window) / 10.0
        self.state.Ki_y = cv2.getTrackbarPos("Ki_y", self.control_window) / 100.0
        self.state.Kd_y = cv2.getTrackbarPos("Kd_y", self.control_window) / 10.0
        
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
        
        # Draw crosshair at center
        center_x, center_y = w // 2, h // 2
        cv2.line(overlay, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 1)
        cv2.line(overlay, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 1)
        cv2.putText(overlay, "Center", (center_x + 5, center_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw target position
        target_x_px = int(center_x + self.state.x_target * self.plate_radius_px / self.plate_radius_m)
        target_y_px = int(center_y - self.state.y_target * self.plate_radius_px / self.plate_radius_m)
        if 0 <= target_x_px < w and 0 <= target_y_px < h:
            cv2.drawMarker(overlay, (target_x_px, target_y_px), (0, 255, 0), 
                          cv2.MARKER_CROSS, 20, 2)
            cv2.putText(overlay, "Target", (target_x_px + 5, target_y_px - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw ball position if available
        if self.state.ball_position is not None:
            x, y = self.state.ball_position
            ball_x_px = int(center_x + x * self.plate_radius_px / self.plate_radius_m)
            ball_y_px = int(center_y - y * self.plate_radius_px / self.plate_radius_m)
            if 0 <= ball_x_px < w and 0 <= ball_y_px < h:
                cv2.circle(overlay, (ball_x_px, ball_y_px), 10, (0, 0, 255), 2)
                cv2.putText(overlay, f"Ball: ({x:.3f}, {y:.3f})m", 
                           (ball_x_px + 15, ball_y_px), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 0, 255), 1)
        
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
        cv2.putText(overlay, f"Kp: ({self.state.Kp_x:.1f}, {self.state.Kp_y:.1f})",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(overlay, f"Ki: ({self.state.Ki_x:.2f}, {self.state.Ki_y:.2f})",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(overlay, f"Kd: ({self.state.Kd_x:.1f}, {self.state.Kd_y:.1f})",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        y_offset = h - 60
        cv2.putText(overlay, "Click to set target | 'q' to quit | 'r' to reset",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return overlay


# ============================================================================
# Main Control Loop
# ============================================================================

class ControlLoop:
    """
    Main control loop that coordinates all components.
    """
    
    def __init__(self, state: ControlState, ball_tracker: BallTracker,
                 normal_controller: NormalController, ui_manager: UIManager):
        """
        Initialize control loop.
        
        Args:
            state: ControlState instance
            ball_tracker: BallTracker instance
            normal_controller: NormalController instance
            ui_manager: UIManager instance
        """
        self.state = state
        self.ball_tracker = ball_tracker
        self.normal_controller = normal_controller
        self.ui_manager = ui_manager
        
        # PID controllers
        self.pid_x = PIDController(
            Kp=state.Kp_x, Ki=state.Ki_x, Kd=state.Kd_x,
            integral_limit=5.0, output_limit=10.0
        )
        self.pid_y = PIDController(
            Kp=state.Kp_y, Ki=state.Ki_y, Kd=state.Kd_y,
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
            
            # Update PID gains from trackbars
            self.ui_manager.update_from_trackbars()
            self.pid_x.set_gains(self.state.Kp_x, self.state.Ki_x, self.state.Kd_x)
            self.pid_y.set_gains(self.state.Kp_y, self.state.Ki_y, self.state.Kd_y)
            self.normal_controller.set_tilt_gain(self.state.tilt_gain)
            self.normal_controller.set_max_tilt_angle(self.state.max_tilt_angle)
            
            # Get ball position
            ball_pos = self.ball_tracker.get_ball_position()
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
                
                # Send to hardware (stub)
                set_desired_normal(n[0], n[1], n[2])
                
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
            frame = self.ball_tracker.get_camera_frame()
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
        set_desired_normal(0.0, 0.0, 1.0)
        cv2.destroyAllWindows()
        self.state.running = False


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    # Initialize components
    state = ControlState()
    ball_tracker = BallTracker()
    normal_controller = NormalController(
        tilt_gain=state.tilt_gain,
        max_tilt_angle_deg=state.max_tilt_angle
    )
    ui_manager = UIManager(
        state=state,
        ball_tracker=ball_tracker,
        plate_radius_m=0.15,  # Adjust based on your platform
        frame_width=640,
        frame_height=480
    )
    
    # Create and run control loop
    control_loop = ControlLoop(state, ball_tracker, normal_controller, ui_manager)
    
    try:
        control_loop.run()
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user")
        state.emergency_stop = True
        set_desired_normal(0.0, 0.0, 1.0)
    except Exception as e:
        print(f"[ERROR] Control loop failed: {e}")
        import traceback
        traceback.print_exc()
        set_desired_normal(0.0, 0.0, 1.0)


if __name__ == "__main__":
    main()

