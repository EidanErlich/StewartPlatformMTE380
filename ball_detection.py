import cv2
import numpy as np
import json
import os

class BallDetector:
    """Computer vision ball detector using HSV color space filtering."""

    def __init__(self, config_file="config.json"):
        """Initialize ball detector with HSV bounds from config file."""
        # Default HSV bounds for orange ball detection
        self.lower_hsv = np.array([5, 150, 150], dtype=np.uint8)
        self.upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
        self.scale_factor = 1.0  # Conversion from normalized coords to meters

        # Load configuration if available
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)

                if 'ball_detection' in config:
                    if config['ball_detection'].get('lower_hsv'):
                        self.lower_hsv = np.array(config['ball_detection']['lower_hsv'], dtype=np.uint8)
                    if config['ball_detection'].get('upper_hsv'):
                        self.upper_hsv = np.array(config['ball_detection']['upper_hsv'], dtype=np.uint8)

                if 'calibration' in config and 'pixel_to_meter_ratio' in config['calibration']:
                    if config['calibration']['pixel_to_meter_ratio']:
                        frame_width = config.get('camera', {}).get('frame_width', 640)
                        self.scale_factor = config['calibration']['pixel_to_meter_ratio'] * (frame_width / 2)

                print(f"[BALL_DETECT] Loaded HSV bounds: {self.lower_hsv} to {self.upper_hsv}")
                print(f"[BALL_DETECT] Scale factor: {self.scale_factor:.6f} m/normalized_unit")

            except Exception as e:
                print(f"[BALL_DETECT] Config load error: {e}, using defaults")
        else:
            print("[BALL_DETECT] No config file found, using default HSV bounds")

    def detect_ball(self, frame):
        """Detect ball in frame and return 2D position in meters.

        Returns:
            found (bool): True if ball detected
            center (tuple): (x, y) pixel coordinates
            radius (float): Ball radius in pixels
            position_m (tuple): (x_m, y_m) ball position in meters
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, None, None, (0.0, 0.0)

        largest = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest)

        if radius < 5 or radius > 100:
            return False, None, None, (0.0, 0.0)

        # Normalize coordinates around image center
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        normalized_x = (x - center_x) / center_x
        normalized_y = (center_y - y) / center_y  # invert Y so up is positive
        x_m = normalized_x * self.scale_factor
        y_m = normalized_y * self.scale_factor

        return True, (int(x), int(y)), radius, (x_m, y_m)

    def draw_detection(self, frame, show_info=True):
        """Draw ball detection overlay with 2D position display."""
        found, center, radius, (x_m, y_m) = self.detect_ball(frame)
        overlay = frame.copy()
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

        # Draw crosshair at frame center
        cv2.line(overlay, (center_x, 0), (center_x, height), (255, 255, 255), 1)
        cv2.line(overlay, (0, center_y), (width, center_y), (255, 255, 255), 1)
        cv2.putText(overlay, "Center", (center_x + 5, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if found:
            cv2.circle(overlay, center, int(radius), (0, 255, 0), 2)
            cv2.circle(overlay, center, 3, (0, 255, 0), -1)

            if show_info:
                cv2.putText(overlay, f"x: {x_m:.4f} m", (center[0] - 50, center[1] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(overlay, f"y: {y_m:.4f} m", (center[0] - 50, center[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return overlay, found, (x_m, y_m)

# --- Legacy 1D compatibility ---
def detect_ball_x(frame):
    """Legacy function: keeps original x-only interface."""
    detector = BallDetector()
    vis_frame, found, (x_m, _) = detector.draw_detection(frame)
    if found:
        x_norm = x_m / detector.scale_factor if detector.scale_factor else 0.0
        x_norm = np.clip(x_norm, -1.0, 1.0)
    else:
        x_norm = 0.0
    return found, x_norm, vis_frame

# --- Testing utility ---
def main():
    detector = BallDetector()
    cap = cv2.VideoCapture(1)

    print("2D Ball Detection Test — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.resize(frame, (640, 480))
        vis_frame, found, (x_m, y_m) = detector.draw_detection(frame)

        if found:
            print(f"Ball detected at x={x_m:.4f} m, y={y_m:.4f} m")

        cv2.imshow("2D Ball Detection", vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
