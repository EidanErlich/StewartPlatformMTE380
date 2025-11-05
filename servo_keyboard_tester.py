#!/usr/bin/env python3
"""
Servo Keyboard Tester
Control three servos individually via keyboard and Arduino serial.

Keys:
  q / w  -> Servo 0: -step / +step
  a / s  -> Servo 1: -step / +step
  z / x  -> Servo 2: -step / +step
  [ / ]  -> Decrease / increase step size
  0      -> Set all servos to 0°
  6      -> Set all servos to 65° (Arduino-constrained max)
  r      -> Reset all to 0°
  t      -> Toggle sweep mode (all motors 0-65° in sync, repeating)
  p      -> Print current angles
  ESC    -> Exit

Notes:
- Arduino firmware constrains angles to 0..65°. This script mirrors that.
- Auto-detects Arduino on macOS (/dev/cu.usbmodem* or /dev/cu.usbserial*).
"""

import sys
import time
from typing import Optional, List

import serial
import serial.tools.list_ports
from pynput import keyboard

ANGLE_MIN = 0
ANGLE_MAX = 65  # must match Arduino's constraint

class ServoKeyboardTester:
    def __init__(self, port: Optional[str] = None, baud: int = 115200):
        self.angles: List[int] = [0, 0, 0]
        self.step: int = 1
        self.ser: Optional[serial.Serial] = None
        self.sweep_active: bool = False
        self.sweep_direction: int = 1  # 1 = going up, -1 = going down

        if port is None:
            port = self.find_arduino_port()
        if not port:
            print("No Arduino found. Specify the port as an argument, e.g. /dev/cu.usbmodemXXXX.")
            sys.exit(1)

        self.connect(port, baud)
        # Initialize servos to current angles (all zeros by default)
        self.send_angles()
        self.print_status(prefix="Initialized")

    def find_arduino_port(self) -> Optional[str]:
        ports = serial.tools.list_ports.comports()
        for p in ports:
            dev = p.device.lower()
            desc = (p.description or "").lower()
            if "usbmodem" in dev or "usbserial" in dev or "arduino" in desc:
                print(f"Found Arduino on: {p.device}")
                return p.device
        return None

    def connect(self, port: str, baud: int) -> None:
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # allow Arduino reset
            # Drain any startup text
            start = time.time()
            while self.ser.in_waiting and time.time() - start < 1.0:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    print(f"Arduino: {line}")
            print(f"Connected to {port} @ {baud} baud")
        except serial.SerialException as e:
            print(f"Failed to open {port}: {e}")
            sys.exit(1)

    def clamp(self, v: int) -> int:
        return max(ANGLE_MIN, min(ANGLE_MAX, int(v)))

    def send_angles(self) -> bool:
        if not (self.ser and self.ser.is_open):
            return False
        cmd = f"{self.angles[0]},{self.angles[1]},{self.angles[2]}\n"
        self.ser.write(cmd.encode("utf-8"))
        time.sleep(0.03)
        ok = False
        # Try to read one line if available
        if self.ser.in_waiting:
            resp = self.ser.readline().decode("utf-8", errors="ignore").strip()
            if resp.startswith("OK:"):
                ok = True
            elif resp.startswith("ERR:"):
                print(f"Arduino error: {resp}")
        return ok

    def bump(self, idx: int, delta: int) -> None:
        self.angles[idx] = self.clamp(self.angles[idx] + delta)
        self.send_angles()
        self.print_status()

    def set_all(self, value: int) -> None:
        v = self.clamp(value)
        self.angles = [v, v, v]
        self.send_angles()
        self.print_status(prefix=f"Set all to {v}°")

    def print_status(self, prefix: Optional[str] = None) -> None:
        pre = f"{prefix} | " if prefix else ""
        sweep_status = " [SWEEP ON]" if self.sweep_active else ""
        print(f"{pre}Angles -> S0: {self.angles[0]:02d}°, S1: {self.angles[1]:02d}°, S2: {self.angles[2]:02d}° | step: {self.step}°{sweep_status}")

    def sweep_step(self) -> None:
        """Move all servos one step in sync during sweep mode"""
        if not self.sweep_active:
            return
        
        # Move all servos together
        target = self.angles[0] + self.sweep_direction
        
        # Check boundaries and reverse direction
        if target > ANGLE_MAX:
            self.sweep_direction = -1
            target = ANGLE_MAX
        elif target < ANGLE_MIN:
            self.sweep_direction = 1
            target = ANGLE_MIN
        
        # Set all servos to same angle
        self.angles = [target, target, target]
        self.send_angles()

    def on_press(self, key) -> Optional[bool]:
        try:
            if key == keyboard.Key.esc:
                print("Exiting...")
                return False
            if isinstance(key, keyboard.KeyCode):
                ch = (key.char or "").lower()
                if ch == 'q':
                    self.bump(0, -self.step)
                elif ch == 'w':
                    self.bump(0, +self.step)
                elif ch == 'a':
                    self.bump(1, -self.step)
                elif ch == 's':
                    self.bump(1, +self.step)
                elif ch == 'z':
                    self.bump(2, -self.step)
                elif ch == 'x':
                    self.bump(2, +self.step)
                elif ch == '[':
                    self.step = max(1, self.step - 1)
                    self.print_status(prefix="Step-")
                elif ch == ']':
                    self.step = min(10, self.step + 1)
                    self.print_status(prefix="Step+")
                elif ch == '0':
                    self.set_all(0)
                elif ch == '6':
                    self.set_all(65)
                elif ch == 'r':
                    self.set_all(0)
                elif ch == 'p':
                    self.print_status(prefix="Status")
                elif ch == 't':
                    # Toggle sweep mode
                    self.sweep_active = not self.sweep_active
                    if self.sweep_active:
                        print("Sweep mode ENABLED - all motors will sweep 0-65° in sync")
                        # Start from current position, determine direction
                        if self.angles[0] >= ANGLE_MAX:
                            self.sweep_direction = -1
                        else:
                            self.sweep_direction = 1
                    else:
                        print("Sweep mode DISABLED")
                    self.print_status()
        except Exception as e:
            print(f"Key handler error: {e}")
        return None

    def run(self) -> None:
        print("\n" + "=" * 60)
        print("Servo Keyboard Tester")
        print("=" * 60)
        print("Controls:")
        print("  q/w  -> Servo0 -/+  |  a/s  -> Servo1 -/+  |  z/x  -> Servo2 -/+ ")
        print("  [ / ] -> step - / +   (1..10 deg)")
        print("  0 -> all 0°   |   6 -> all 65°   |   r -> reset all 0°   |   p -> print")
        print("  t -> toggle sweep mode (all motors 0-65° in sync, repeating)")
        print("  ESC to exit")
        print("=" * 60)
        # Start listener in non-blocking mode
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        
        try:
            # Main loop for sweep functionality
            while listener.running:
                if self.sweep_active:
                    self.sweep_step()
                    time.sleep(0.05)  # Sweep speed control (50ms per step)
                else:
                    time.sleep(0.1)  # Reduce CPU usage when not sweeping
        except KeyboardInterrupt:
            pass
        finally:
            listener.stop()
            if self.ser and self.ser.is_open:
                self.ser.close()
                print("Serial closed.")


def main():
    port = None
    if len(sys.argv) > 1:
        port = sys.argv[1]
    tester = ServoKeyboardTester(port=port)
    tester.run()


if __name__ == "__main__":
    main()
