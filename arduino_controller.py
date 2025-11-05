#!/usr/bin/env python3
"""
Arduino Stewart Platform Controller
Controls a 3-servo Stewart platform via serial communication with Arduino.
Use arrow keys to tilt the platform.
"""

import sys
import numpy as np
import serial
import serial.tools.list_ports
import time
from pynput import keyboard

# Import the StewartPlatform class from inverseKinematics
from inverseKinematics import StewartPlatform, SERVO_HORN_LENGTH, ROD_LENGTH

class ArduinoStewartController:
    def __init__(self, port=None, baud=115200):
        self.platform = StewartPlatform()
        self.normal = np.array([0, 0, 1], dtype=float)
        self.ser = None
        self.max_tilt = np.radians(30)  # 30 degree max tilt
        self.step = 0.1
        
        # Find and connect to Arduino
        if port is None:
            port = self.find_arduino_port()
        
        if port:
            self.connect(port, baud)
        else:
            print("No Arduino found. Please specify port manually.")
            sys.exit(1)
    
    def find_arduino_port(self):
        """Auto-detect Arduino port"""
        ports = serial.tools.list_ports.comports()
        for p in ports:
            # Look for common Arduino USB identifiers
            if 'usbmodem' in p.device or 'usbserial' in p.device or 'Arduino' in str(p.description):
                print(f"Found Arduino on: {p.device}")
                return p.device
        return None
    
    def connect(self, port, baud):
        """Connect to Arduino via serial"""
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            
            # Read startup messages
            while self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"Arduino: {line}")
            
            print(f"Connected to Arduino on {port} at {baud} baud")
            print("Initializing all servos to 0 degrees...")
            # Send all servos to 0 degrees directly
            self.send_angles([0, 0, 0])
            print("Servos initialized: [0°, 0°, 0°]")
            
        except serial.SerialException as e:
            print(f"Failed to connect to {port}: {e}")
            sys.exit(1)
    
    def send_angles(self, angles_deg):
        """Send servo angles to Arduino in degrees"""
        if self.ser and self.ser.is_open:
            # Format: "angle0,angle1,angle2\n"
            command = f"{int(angles_deg[0])},{int(angles_deg[1])},{int(angles_deg[2])}\n"
            self.ser.write(command.encode())
            
            # Read response
            time.sleep(0.05)
            if self.ser.in_waiting:
                response = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if response.startswith('OK:'):
                    return True
                elif response.startswith('ERR:'):
                    print(f"Arduino error: {response}")
                    return False
        return False
    
    def update_platform(self):
        """Calculate servo angles and send to Arduino"""
        # Calculate servo command angles (deg) with zero offsets and clamp to 0..65
        angles_deg = self.platform.calculate_servo_angles(
            self.normal,
            degrees=True,
            apply_offsets=True,
            clamp_min=0.0,
            clamp_max=65.0,
        )
        
        # Send to Arduino
        success = self.send_angles(angles_deg)
        
        if success:
            print(f"Normal: [{self.normal[0]:.3f}, {self.normal[1]:.3f}, {self.normal[2]:.3f}] | "
                  f"Angles: [{angles_deg[0]:.1f}°, {angles_deg[1]:.1f}°, {angles_deg[2]:.1f}°]")
        
        return success
    
    def apply_tilt_limit(self):
        """Enforce maximum tilt angle"""
        z_axis = np.array([0, 0, 1])
        tilt_angle = np.arccos(np.clip(np.dot(self.normal, z_axis), -1, 1))
        
        if tilt_angle > self.max_tilt:
            # Project back to max tilt
            tilt_direction = np.array([self.normal[0], self.normal[1], 0])
            if np.linalg.norm(tilt_direction) > 1e-6:
                tilt_direction = tilt_direction / np.linalg.norm(tilt_direction)
                self.normal = np.array([
                    tilt_direction[0] * np.sin(self.max_tilt),
                    tilt_direction[1] * np.sin(self.max_tilt),
                    np.cos(self.max_tilt)
                ])
                self.normal = self.normal / np.linalg.norm(self.normal)
    
    def on_press(self, key):
        """Handle keyboard input"""
        try:
            if key == keyboard.Key.up:
                self.normal[1] += self.step
            elif key == keyboard.Key.down:
                self.normal[1] -= self.step
            elif key == keyboard.Key.left:
                self.normal[0] -= self.step
            elif key == keyboard.Key.right:
                self.normal[0] += self.step
            elif key == keyboard.KeyCode.from_char('r') or key == keyboard.KeyCode.from_char('R'):
                self.normal = np.array([0, 0, 1], dtype=float)
                print("Reset to neutral position")
            elif key == keyboard.Key.esc:
                print("\nExiting...")
                return False
            else:
                return True
            
            # Normalize
            if np.linalg.norm(self.normal) < 1e-6:
                self.normal = np.array([0, 0, 1], dtype=float)
            else:
                self.normal = self.normal / np.linalg.norm(self.normal)
            
            # Apply tilt limit
            self.apply_tilt_limit()
            
            # Update platform
            self.update_platform()
            
        except AttributeError:
            pass
        
        return True
    
    def run(self):
        """Start the interactive controller"""
        print("\n" + "="*60)
        print("Stewart Platform Arduino Controller")
        print("="*60)
        print("Controls:")
        print("  ↑/↓  - Tilt forward/backward")
        print("  ←/→  - Tilt left/right")
        print("  R    - Reset to neutral")
        print("  ESC  - Exit")
        print("="*60)
        print(f"Max tilt: {np.degrees(self.max_tilt):.1f}°")
        print("Ready! Use arrow keys to control the platform.\n")
        
        # Start keyboard listener
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()
        
        # Close serial connection
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial connection closed.")


def main():
    # Check for command line port argument
    port = None
    if len(sys.argv) > 1:
        port = sys.argv[1]
    
    try:
        controller = ArduinoStewartController(port=port)
        controller.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
