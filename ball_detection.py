import cv2
import numpy as np
import json
import os

class BallDetector:
    """Computer vision ball detector using HSV color space filtering with camera calibration."""

    def __init__(self, config_file="config.json", calib_file="cal/camera_calib.npz"):
        """Initialize ball detector with HSV bounds and camera calibration.
        
        Args:
            config_file: Path to config.json with HSV bounds and coordinate frame
            calib_file: Path to camera calibration .npz file with K and dist
        """
        # Default HSV bounds for orange ball detection
        self.lower_hsv = np.array([5, 150, 150], dtype=np.uint8)
        self.upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
        
        # Coordinate frame parameters
        self.origin_px = None
        self.x_axis = None
        self.y_axis = None
        self.pixel_to_meter_ratio = None
        self.has_coordinate_frame = False
        
        # Camera calibration parameters
        self.K = None
        self.dist = None
        self.newK = None
        self.has_camera_calib = False

        # Load camera calibration
        if os.path.exists(calib_file):
            try:
                data = np.load(calib_file, allow_pickle=True)
                self.K = data["K"].astype(np.float64)
                self.dist = data["dist"].astype(np.float64)
                self.has_camera_calib = True
                print(f"[BALL_DETECT] Loaded camera calibration from {calib_file}")
                print(f"  K shape: {self.K.shape}, dist shape: {self.dist.shape}")
            except Exception as e:
                print(f"[BALL_DETECT] Camera calibration load error: {e}")
        else:
            print(f"[BALL_DETECT] Camera calibration file not found: {calib_file}")

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

                # Load coordinate frame
                if 'coordinate_frame' in config:
                    frame = config['coordinate_frame']
                    if 'origin_px' in frame and 'x_axis' in frame and 'y_axis' in frame:
                        self.origin_px = np.array(frame['origin_px'], dtype=np.float64)
                        self.x_axis = np.array(frame['x_axis'], dtype=np.float64)
                        self.y_axis = np.array(frame['y_axis'], dtype=np.float64)
                        self.has_coordinate_frame = True
                        print(f"[BALL_DETECT] Loaded coordinate frame:")
                        print(f"  Origin: {self.origin_px}")
                        print(f"  X-axis: {self.x_axis}")
                        print(f"  Y-axis: {self.y_axis}")

                # Load pixel to meter ratio from platform calibration
                if 'platform' in config and 'pixel_to_meter_ratio' in config['platform']:
                    self.pixel_to_meter_ratio = config['platform']['pixel_to_meter_ratio']
                    print(f"[BALL_DETECT] Pixel to meter ratio: {self.pixel_to_meter_ratio:.6f} m/px")

                print(f"[BALL_DETECT] Loaded HSV bounds: {self.lower_hsv} to {self.upper_hsv}")

            except Exception as e:
                print(f"[BALL_DETECT] Config load error: {e}, using defaults")
        else:
            print("[BALL_DETECT] No config file found, using default HSV bounds")

    def detect_ball(self, frame, undistort=True):
        """Detect ball in frame and return 2D position in meters using calibrated coordinate frame.

        Args:
            frame: Input frame (BGR format)
            undistort: Whether to apply camera undistortion (default: True)

        Returns:
            found (bool): True if ball detected
            center (tuple): (x, y) pixel coordinates in undistorted frame
            radius (float): Ball radius in pixels
            position_m (tuple): (x_m, y_m) ball position in meters in the calibrated coordinate frame
        """
        # Undistort frame if camera calibration is available and requested
        if undistort and self.has_camera_calib:
            if self.newK is None:
                # Compute optimal new camera matrix once
                h, w = frame.shape[:2]
                self.newK, roi = cv2.getOptimalNewCameraMatrix(
                    self.K, self.dist, (w, h), 1, (w, h)
                )
            frame = cv2.undistort(frame, self.K, self.dist, None, self.newK)
        
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

        # Transform to calibrated coordinate frame if available
        if self.has_coordinate_frame and self.pixel_to_meter_ratio is not None:
            # Vector from origin to ball in pixel coordinates
            ball_px = np.array([x, y], dtype=np.float64)
            delta_px = ball_px - self.origin_px
            
            # Project onto calibrated axes
            x_m = np.dot(delta_px, self.x_axis) * self.pixel_to_meter_ratio
            y_m = np.dot(delta_px, self.y_axis) * self.pixel_to_meter_ratio
        else:
            # Fallback: use normalized coordinates around image center
            center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
            normalized_x = (x - center_x) / center_x
            normalized_y = (center_y - y) / center_y  # invert Y so up is positive
            x_m = normalized_x * 0.15  # Assume 0.15m radius
            y_m = normalized_y * 0.15

        return True, (int(x), int(y)), radius, (x_m, y_m)

    def draw_detection(self, frame, show_info=True):
        """Draw ball detection overlay with 2D position display in calibrated coordinate frame.
        
        Note: Frame will be undistorted if camera calibration is available.
        
        Returns:
            overlay: Annotated frame showing detection results
            found: True if ball was detected
            position: (x_m, y_m) position in meters
        """
        # Undistort frame if camera calibration is available
        if self.has_camera_calib:
            if self.newK is None:
                # Compute optimal new camera matrix once
                h, w = frame.shape[:2]
                self.newK, roi = cv2.getOptimalNewCameraMatrix(
                    self.K, self.dist, (w, h), 1, (w, h)
                )
            frame = cv2.undistort(frame, self.K, self.dist, None, self.newK)
        
        # Detect ball on the undistorted frame (pass undistort=False to avoid double undistortion)
        found, center, radius, (x_m, y_m) = self.detect_ball(frame, undistort=False)
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Draw calibrated origin if available
        if self.has_coordinate_frame and self.origin_px is not None:
            origin = (int(self.origin_px[0]), int(self.origin_px[1]))
            cv2.circle(overlay, origin, 5, (0, 255, 255), -1)
            cv2.putText(overlay, "Origin", (origin[0] + 10, origin[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Draw coordinate axes
            axis_length = 100
            x_end = (int(origin[0] + self.x_axis[0] * axis_length),
                     int(origin[1] + self.x_axis[1] * axis_length))
            y_end = (int(origin[0] + self.y_axis[0] * axis_length),
                     int(origin[1] + self.y_axis[1] * axis_length))
            
            cv2.arrowedLine(overlay, origin, x_end, (0, 0, 255), 2, tipLength=0.3)  # Red for X
            cv2.arrowedLine(overlay, origin, y_end, (0, 255, 0), 2, tipLength=0.3)  # Green for Y
            cv2.putText(overlay, "X", (x_end[0] + 5, x_end[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(overlay, "Y", (y_end[0] + 5, y_end[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Fallback: draw crosshair at frame center
            center_x, center_y = width // 2, height // 2
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

# --- Testing utility ---
def main():
    detector = BallDetector()
    cap = cv2.VideoCapture(0)

    print("2D Ball Detection Test — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

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
