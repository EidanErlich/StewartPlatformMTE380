"""
Stewart Platform Calibration Script
-----------------------------------
Calibrates:
 - Ball HSV color range
 - Platform center and diameter
 - Pixel-to-meter ratio (using input platform radius)
Saves: config.json
"""

import cv2
import numpy as np
import json
from datetime import datetime

class StewartPlatformCalibrator:
    def __init__(self):
        self.CAM_INDEX = 1
        self.FRAME_W, self.FRAME_H = 640, 480
        self.current_frame = None

        # Ball color calibration
        self.hsv_samples = []
        self.lower_hsv = None
        self.upper_hsv = None

        # Platform geometry
        self.platform_center_px = None
        self.platform_diameter_px = None
        self.pixel_to_meter_ratio = None
        self.platform_radius_m = None

        # Phases
        self.phase = "color"  # "color" -> "platform" -> "complete"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.phase == "color":
            self.sample_color(x, y)

    def sample_color(self, x, y):
        """Sample 5x5 HSV region around clicked point for ball color."""
        if self.current_frame is None:
            return

        hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
        region = hsv[max(0, y-2):y+3, max(0, x-2):x+3]
        samples = region.reshape(-1, 3)
        self.hsv_samples.extend(samples)

        samples = np.array(self.hsv_samples)
        h_margin, s_margin, v_margin = 10, 30, 30
        self.lower_hsv = np.maximum(np.min(samples, axis=0) - [h_margin, s_margin, v_margin], [0, 0, 0])
        self.upper_hsv = np.minimum(np.max(samples, axis=0) + [h_margin, s_margin, v_margin], [179, 255, 255])
        print(f"[COLOR] Samples: {len(self.hsv_samples)}")

    def detect_platform_circle(self, frame):
        """Detect circular platform using Hough Circle Transform."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                   param1=80, param2=30, minRadius=80, maxRadius=300)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            x, y, r = circles[0][0]
            self.platform_center_px = (int(x), int(y))
            self.platform_diameter_px = 2 * int(r)
            return (x, y, r)
        return None

    def save_config(self):
        config = {
            "timestamp": datetime.now().isoformat(),
            "camera": {
                "index": int(self.CAM_INDEX),
                "frame_width": int(self.FRAME_W),
                "frame_height": int(self.FRAME_H)
            },
            "platform": {
                "center_px": self.platform_center_px,
                "diameter_px": self.platform_diameter_px,
                "radius_m": self.platform_radius_m,
                "pixel_to_meter_ratio": self.pixel_to_meter_ratio
            },
            "ball_detection": {
                "lower_hsv": self.lower_hsv.tolist() if self.lower_hsv is not None else None,
                "upper_hsv": self.upper_hsv.tolist() if self.upper_hsv is not None else None
            }
        }
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("[SAVE] Configuration saved to config.json")

    def run(self):
        cap = cv2.VideoCapture(self.CAM_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_H)
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.mouse_callback)

        print("[INFO] Stewart Platform Calibration")
        print("Phase 1: Click on the ball to sample color. Press 'c' when done.")
        print("Phase 2: Ensure platform is visible. Press 'p' to detect the platform.")
        print("Enter platform radius (m) when prompted.")
        print("Press 's' to save. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            self.current_frame = frame.copy()

            display = frame.copy()

            # Show detection overlays
            if self.lower_hsv is not None:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    ((x, y), r) = cv2.minEnclosingCircle(c)
                    if r > 5:
                        cv2.circle(display, (int(x), int(y)), int(r), (0, 255, 255), 2)

            if self.platform_center_px and self.platform_diameter_px:
                cv2.circle(display, self.platform_center_px, self.platform_diameter_px // 2, (255, 0, 0), 2)
                cv2.circle(display, self.platform_center_px, 5, (0, 255, 0), -1)

            cv2.putText(display, f"Phase: {self.phase}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and self.phase == "color":
                self.phase = "platform"
                print("[INFO] Color calibration done. Press 'p' to detect platform.")
            elif key == ord('p') and self.phase == "platform":
                result = self.detect_platform_circle(frame)
                if result:
                    x, y, r = result
                    print(f"[PLATFORM] Detected center=({x},{y}), radius={r}px")
                    self.platform_radius_m = float(input("Enter platform radius in meters: "))
                    self.pixel_to_meter_ratio = self.platform_radius_m / r
                    print(f"[CAL] Pixel-to-meter ratio: {self.pixel_to_meter_ratio:.6f}")
                    self.phase = "complete"
                else:
                    print("[WARN] No circle detected. Adjust lighting or view.")
            elif key == ord('s') and self.phase == "complete":
                self.save_config()
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrator = StewartPlatformCalibrator()
    calibrator.run()
