#!/usr/bin/env python3
"""
Stewart Platform 1D PID Controller with GUI
Real-time PID tuning interface for single-servo ball balancing.
"""

import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from threading import Thread
import queue
from ball_detection import BallDetector


class StewartPIDController:
    def __init__(self, control_servo=0):
        """Initialize controller with GUI-tunable PID gains."""
        # PID gains (controlled by sliders in GUI)
        self.Kp = 15.0
        self.Ki = 0.0
        self.Kd = 3.0
        
        # Servo configuration
        self.control_servo = control_servo  # Which servo to control (0, 1, or 2)
        self.servo_angles = [40, 40, 40]  # All servos start at neutral (raised)
        self.neutral_angle = 40  # Neutral position (raised to allow both directions)
        self.max_angle_deviation = 25  # Max deviation from neutral in degrees
        
        # Arduino connection
        self.ser = None
        self.arduino_port = None
        
        # Ball detection
        self.detector = BallDetector()
        
        # Controller state
        self.setpoint = 0.0  # Target position (meters)
        self.integral = 0.0
        self.prev_error = 0.0
        
        # Data logs for plotting
        self.time_log = []
        self.position_log = []
        self.setpoint_log = []
        self.control_log = []
        self.servo_angle_log = []
        self.start_time = None
        
        # Thread-safe queue for ball position
        self.position_queue = queue.Queue(maxsize=1)
        self.running = False
        
    def find_arduino_port(self):
        """Auto-detect Arduino port"""
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if 'usbmodem' in p.device or 'usbserial' in p.device or 'Arduino' in str(p.description):
                print(f"[ARDUINO] Found on: {p.device}")
                return p.device
        return None
    
    def connect_arduino(self):
        """Connect to Arduino via serial"""
        self.arduino_port = self.find_arduino_port()
        if not self.arduino_port:
            print("[ARDUINO] Not found - running in simulation mode")
            return False
        
        try:
            self.ser = serial.Serial(self.arduino_port, 115200, timeout=1)
            time.sleep(2)  # Wait for Arduino reset
            
            # Clear startup messages
            while self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"[ARDUINO] {line}")
            
            print(f"[ARDUINO] Connected on {self.arduino_port}")
            # Initialize to neutral (40 degrees)
            self.send_angles([40, 40, 40])
            return True
        except Exception as e:
            print(f"[ARDUINO] Connection failed: {e}")
            return False
    
    def send_angles(self, angles_deg):
        """Send servo angles to Arduino"""
        if self.ser and self.ser.is_open:
            command = f"{int(angles_deg[0])},{int(angles_deg[1])},{int(angles_deg[2])}\n"
            try:
                self.ser.write(command.encode())
                time.sleep(0.03)
                
                if self.ser.in_waiting:
                    response = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if response.startswith('ERR:'):
                        print(f"[ARDUINO] Error: {response}")
            except Exception as e:
                print(f"[ARDUINO] Send failed: {e}")
    
    def update_servo(self, angle_deg):
        """Update only the controlled servo"""
        clamped_angle = np.clip(angle_deg, 0, 65)
        self.servo_angles[self.control_servo] = clamped_angle
        self.send_angles(self.servo_angles)
    
    def reset_platform(self):
        """Reset all servos to neutral"""
        self.servo_angles = [self.neutral_angle] * 3
        self.send_angles(self.servo_angles)
        print("[RESET] Platform reset to neutral")
    
    def update_pid(self, position, dt=0.033):
        """Perform PID calculation and return control output"""
        error = self.setpoint - position
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term (with anti-windup)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10.0, 10.0)
        I = self.Ki * self.integral
        
        # Derivative term
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        D = self.Kd * derivative
        self.prev_error = error
        
        # PID output (convert to angle change)
        output = P + I + D
        
        return output
    
    def camera_thread(self):
        """Video capture and ball detection thread"""
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Detect ball
            vis_frame, found, (x_m, y_m) = self.detector.draw_detection(frame, show_info=True)
            
            if found:
                # Use Y-axis for control
                position_m = y_m
                # Update position queue (keep only latest)
                try:
                    if self.position_queue.full():
                        self.position_queue.get_nowait()
                    self.position_queue.put_nowait(position_m)
                except:
                    pass
            
            # Add status overlay
            cv2.putText(vis_frame, f"Control Servo: {self.control_servo}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Servo Angle: {self.servo_angles[self.control_servo]:.1f}deg", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Kp:{self.Kp:.1f} Ki:{self.Ki:.1f} Kd:{self.Kd:.1f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("Stewart Platform Ball Tracking", vis_frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                self.running = False
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def control_thread(self):
        """PID control loop thread"""
        if not self.connect_arduino():
            print("[WARNING] No Arduino - running in simulation mode")
        
        self.start_time = time.time()
        last_time = self.start_time
        
        while self.running:
            try:
                # Get latest ball position
                position = self.position_queue.get(timeout=0.1)
                
                # Calculate dt
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                
                # Compute PID control output
                control_output = self.update_pid(position, dt)
                
                # Convert to servo angle (INVERTED for correct direction)
                target_angle = self.neutral_angle - control_output
                
                # Clamp to safe range
                target_angle = np.clip(target_angle,
                                      self.neutral_angle - self.max_angle_deviation,
                                      self.neutral_angle + self.max_angle_deviation)
                
                # Send to servo
                self.update_servo(target_angle)
                
                # Log data
                elapsed_time = current_time - self.start_time
                self.time_log.append(elapsed_time)
                self.position_log.append(position)
                self.setpoint_log.append(self.setpoint)
                self.control_log.append(control_output)
                self.servo_angle_log.append(self.servo_angles[self.control_servo])
                
                print(f"Pos: {position:+.4f}m | Error: {self.setpoint - position:+.4f} | "
                      f"Output: {control_output:+.2f} | Angle: {self.servo_angles[self.control_servo]:.1f}°")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[CONTROL] Error: {e}")
                break
        
        # Cleanup
        self.reset_platform()
        if self.ser and self.ser.is_open:
            self.ser.close()
    
    def create_gui(self):
        """Build Tkinter GUI with PID tuning sliders"""
        self.root = tk.Tk()
        self.root.title(f"Stewart Platform PID Controller - Servo {self.control_servo}")
        self.root.geometry("600x500")
        
        # Title
        ttk.Label(self.root, text="Stewart Platform 1D PID Tuning", 
                 font=("Arial", 18, "bold")).pack(pady=10)
        
        # Servo selection frame
        servo_frame = ttk.LabelFrame(self.root, text="Servo Selection", padding=10)
        servo_frame.pack(pady=5, padx=20, fill=tk.X)
        
        self.servo_var = tk.IntVar(value=self.control_servo)
        for i in range(3):
            ttk.Radiobutton(servo_frame, text=f"Servo {i}", variable=self.servo_var,
                          value=i, command=self.change_servo).pack(side=tk.LEFT, padx=10)
        
        # Kp slider
        ttk.Label(self.root, text="Kp (Proportional)", font=("Arial", 12)).pack(pady=(10,0))
        self.kp_var = tk.DoubleVar(value=self.Kp)
        kp_slider = ttk.Scale(self.root, from_=0, to=50, variable=self.kp_var,
                             orient=tk.HORIZONTAL, length=550)
        kp_slider.pack(pady=5)
        self.kp_label = ttk.Label(self.root, text=f"Kp: {self.Kp:.1f}", font=("Arial", 11))
        self.kp_label.pack()
        
        # Ki slider
        ttk.Label(self.root, text="Ki (Integral)", font=("Arial", 12)).pack(pady=(10,0))
        self.ki_var = tk.DoubleVar(value=self.Ki)
        ki_slider = ttk.Scale(self.root, from_=0, to=5, variable=self.ki_var,
                             orient=tk.HORIZONTAL, length=550)
        ki_slider.pack(pady=5)
        self.ki_label = ttk.Label(self.root, text=f"Ki: {self.Ki:.2f}", font=("Arial", 11))
        self.ki_label.pack()
        
        # Kd slider
        ttk.Label(self.root, text="Kd (Derivative)", font=("Arial", 12)).pack(pady=(10,0))
        self.kd_var = tk.DoubleVar(value=self.Kd)
        kd_slider = ttk.Scale(self.root, from_=0, to=10, variable=self.kd_var,
                             orient=tk.HORIZONTAL, length=550)
        kd_slider.pack(pady=5)
        self.kd_label = ttk.Label(self.root, text=f"Kd: {self.Kd:.2f}", font=("Arial", 11))
        self.kd_label.pack()
        
        # Setpoint slider
        ttk.Label(self.root, text="Setpoint (meters)", font=("Arial", 12)).pack(pady=(10,0))
        self.setpoint_var = tk.DoubleVar(value=self.setpoint)
        setpoint_slider = ttk.Scale(self.root, from_=-0.1, to=0.1,
                                    variable=self.setpoint_var,
                                    orient=tk.HORIZONTAL, length=550)
        setpoint_slider.pack(pady=5)
        self.setpoint_label = ttk.Label(self.root, text=f"Setpoint: {self.setpoint:.3f}m", 
                                       font=("Arial", 11))
        self.setpoint_label.pack()
        
        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=20)
        ttk.Button(button_frame, text="Reset Integral",
                  command=self.reset_integral).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset Platform",
                  command=self.reset_platform).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Plot Results",
                  command=self.plot_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop",
                  command=self.stop).pack(side=tk.LEFT, padx=5)
        
        # Start GUI update loop
        self.update_gui()
    
    def change_servo(self):
        """Handle servo selection change"""
        new_servo = self.servo_var.get()
        if new_servo != self.control_servo:
            self.control_servo = new_servo
            self.reset_platform()
            self.root.title(f"Stewart Platform PID Controller - Servo {self.control_servo}")
            print(f"[SERVO] Switched to Servo {self.control_servo}")
    
    def update_gui(self):
        """Update GUI with current values"""
        if self.running:
            # Update PID parameters from sliders
            self.Kp = self.kp_var.get()
            self.Ki = self.ki_var.get()
            self.Kd = self.kd_var.get()
            self.setpoint = self.setpoint_var.get()
            
            # Update labels
            self.kp_label.config(text=f"Kp: {self.Kp:.1f}")
            self.ki_label.config(text=f"Ki: {self.Ki:.2f}")
            self.kd_label.config(text=f"Kd: {self.Kd:.2f}")
            self.setpoint_label.config(text=f"Setpoint: {self.setpoint:.3f}m")
            
            # Schedule next update
            self.root.after(50, self.update_gui)
    
    def reset_integral(self):
        """Clear integral error"""
        self.integral = 0.0
        print("[RESET] Integral term cleared")
    
    def plot_results(self):
        """Plot position and control data"""
        if not self.time_log:
            print("[PLOT] No data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Ball position
        ax1.plot(self.time_log, self.position_log, label="Ball Position", linewidth=2)
        ax1.plot(self.time_log, self.setpoint_log, label="Setpoint",
                linestyle="--", linewidth=2, color='red')
        ax1.set_ylabel("Position (m)")
        ax1.set_title(f"Stewart Platform 1D Control - Servo {self.control_servo} "
                     f"(Kp={self.Kp:.1f}, Ki={self.Ki:.2f}, Kd={self.Kd:.2f})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Servo angle
        ax2.plot(self.time_log, self.servo_angle_log, label="Servo Angle",
                color="orange", linewidth=2)
        ax2.axhline(y=self.neutral_angle, color='gray', linestyle='--', 
                   label='Neutral', alpha=0.7)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Servo Angle (degrees)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def stop(self):
        """Stop controller and cleanup"""
        self.running = False
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass
    
    def run(self):
        """Main entry point"""
        print("="*60)
        print("Stewart Platform 1D PID Controller with GUI")
        print("="*60)
        print(f"Controlling: Servo {self.control_servo}")
        print("Use sliders to tune PID gains in real-time")
        print("Press ESC in camera window to exit")
        print("="*60)
        
        self.running = True
        
        # Start threads
        cam_thread = Thread(target=self.camera_thread, daemon=True)
        ctrl_thread = Thread(target=self.control_thread, daemon=True)
        cam_thread.start()
        ctrl_thread.start()
        
        # Run GUI
        self.create_gui()
        self.root.mainloop()
        
        # Cleanup
        self.running = False
        print("[INFO] Controller stopped")


def main():
    import sys
    control_servo = 0
    
    if len(sys.argv) > 1:
        control_servo = int(sys.argv[1])
    
    try:
        controller = StewartPIDController(control_servo=control_servo)
        controller.run()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
