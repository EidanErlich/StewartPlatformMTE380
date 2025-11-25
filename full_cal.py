#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import json
from datetime import datetime

# ---- FIXED PLATE GEOMETRY (your setup) ----
ANGLES_DEG = [0.0, 120.0, 240.0]  # evenly spaced, CCW from +X

# ---- RUNTIME CONFIG (hardcoded, not CLI) ----
# Only --calib and --cam are accepted on the command line; everything else is fixed here.
PREVIEW = False                 # undistort preview window
JSON_OUTPUT = False             # print one-line JSON each frame
MAX_PAIR_PX = None              # optional max pixel distance to accept screw pairs
POINT_DIAMETER_M = 0.01        # expected point diameter in meters (1cm)
WORKING_DISTANCE_M = 0.3       # estimated working distance in meters
POINT_SMOOTH_ALPHA = 0.3       # point smoothing factor (0-1)
PAIR_SEP_M = 0.07              # known physical separation (meters) between two screws in a pair (7cm)
PAIR_SEP_TOL = 0.3             # tolerance fraction for PAIR_SEP_M when filtering pairs (±30%)
MIN_CIRCULARITY = 0.7          # minimum circularity for screw blob detection (0-1, higher = more circular)

# Coordinate frame storage (updated when motors stabilized)
coordinate_frame = None

"""
FOR HUMAN

Motor A - Connection 0
Motor B - Connection 2
Motor C - Connection 4
"""


def estimate_pixel_area_for_size(physical_diameter_m, K, working_distance_m=0.5):
    """
    Estimate pixel area for a physical object size.
    physical_diameter_m: diameter in meters (e.g., 0.01 for 1cm)
    K: camera matrix
    working_distance_m: estimated working distance in meters
    Returns: estimated pixel area
    """
    fx = K[0, 0]
    # Pixel size = (physical_size * focal_length) / distance
    pixel_diameter = (physical_diameter_m * fx) / working_distance_m
    pixel_radius = pixel_diameter / 2.0
    pixel_area = np.pi * pixel_radius ** 2
    return pixel_area


def load_camera_calib(calib_path="cal/camera_calib.npz"):
    """Load camera intrinsics from an npz file."""
    data = np.load(calib_path, allow_pickle=True)
    K = data["K"].astype(np.float64)
    dist = data["dist"].astype(np.float64)
    return K, dist


class StewartPlatformCalibrator:
    def __init__(self):
        self.CAM_INDEX = 0
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
        # If a coordinate_frame was computed externally (origin, axes), include it in the saved config
        global coordinate_frame
        if coordinate_frame is not None:
            config["coordinate_frame"] = {
                'origin_px': coordinate_frame.get('origin_px'),
                'x_axis': coordinate_frame.get('x_axis'),
                'y_axis': coordinate_frame.get('y_axis'),
                'timestamp': coordinate_frame.get('timestamp')
            }
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("[SAVE] Configuration saved to config.json")

    def run(self):
        cap = cv2.VideoCapture(self.CAM_INDEX)
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.mouse_callback)

        print("[INFO] Stewart Platform Calibration")
        print("Phase 1: Click on the ball to sample color. Press 'c' when done.")
        print("Phase 2: Ensure platform is visible. Press 'p' to detect the platform radius.")
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
                # For this flow we expect the coordinate_frame (origin & axes) to have been
                # computed beforehand (outside this class). We still use Hough to get a
                # reliable pixel radius (r) but we DO NOT use the Hough center; instead
                # we project the externally-computed origin into image pixels and use
                # that as the platform center.
                global coordinate_frame
                if coordinate_frame is None:
                    print("[WARN] No precomputed coordinate frame found. Compute origin first, then re-run this step.")
                    continue

                # try to detect a circle to obtain a pixel radius (we'll override center)
                result = self.detect_platform_circle(frame)
                if result is None:
                    print("[WARN] Could not detect a platform circle to obtain pixel radius. Adjust view or lighting.")
                    continue

                _, _, r = result
                print(f"[PLATFORM] Detected radius={r}px (Hough). Using precomputed origin as center.)")

                # ask user for real-world platform radius in meters (keeps previous UX)
                try:
                    self.platform_radius_m = float(input("Enter platform radius in meters: "))
                except Exception:
                    print("Invalid radius input; try again.")
                    continue

                # compute pixel-to-meter ratio from measured pixel radius
                try:
                    self.pixel_to_meter_ratio = float(self.platform_radius_m) / float(r)
                    self.platform_diameter_px = int(round(2.0 * float(r)))
                except Exception:
                    print("[ERROR] Could not compute pixel-to-meter ratio from radius. Check inputs.")
                    continue

                # Project the precomputed origin (in pixel coords) to use as platform center
                try:
                    K, _ = load_camera_calib()
                    origin_px = np.array(coordinate_frame.get('origin_px'), dtype=np.float64)
                    self.platform_center_px = (int(round(origin_px[0])), int(round(origin_px[1])))
                    print(f"[PLATFORM] Using origin as platform center: ({self.platform_center_px[0]},{self.platform_center_px[1]})")
                    self.phase = "complete"
                except Exception as ex:
                    print(f"[ERROR] Failed using origin as platform center: {ex}")
            elif key == ord('s') and self.phase == "complete":
                self.save_config()
                break

        cap.release()
        cv2.destroyAllWindows()


def detect_pairs_from_frame(frame, K, detector=None, point_diameter_m=POINT_DIAMETER_M,
                            working_distance_m=WORKING_DISTANCE_M, max_pair_px=MAX_PAIR_PX):
    """
    Detect dark circular blobs in the given frame and return paired blob points.
    Returns a tuple (img_pts, pairs) where img_pts is an (2*N,2) array containing
    the two blob points per pair in the same order as pairs, and pairs is a list of
    index tuples into the original pts array indicating the paired points.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if detector is None:
        # estimate reasonable area range from camera intrinsics and expected size
        est_area = estimate_pixel_area_for_size(point_diameter_m, K, working_distance_m)
        min_area = int(max(5, est_area * 0.3))
        max_area = int(max(20, est_area * 2.0))
        detector = make_blob_detector(min_area=min_area, max_area=max_area, dark=True)

    keypoints = detector.detect(blurred)

    # Filter keypoints by approximate diameter derived from K and working distance
    fx = K[0, 0]
    expected_diameter_px = (point_diameter_m * fx) / working_distance_m
    min_diameter_px = expected_diameter_px * 0.5
    max_diameter_px = expected_diameter_px * 1.5

    filtered_keypoints = []
    for kp in keypoints:
        if min_diameter_px <= kp.size <= max_diameter_px:
            filtered_keypoints.append(kp)
    keypoints = filtered_keypoints

    if keypoints:
        refined_pts = [refine_centroid_subpixel(gray, kp.pt, win=11) for kp in keypoints]
        pts = np.array(refined_pts, dtype=np.float32)
    else:
        pts = np.empty((0, 2), dtype=np.float32)

    pairs = []
    img_pts = None
    if len(pts) >= 2:
        candidate_pairs = pair_points_by_distance(pts, expected_pairs=(len(pts) // 2), max_pair_px=max_pair_px)
        if candidate_pairs:
            used_idx = set()
            for a, b in candidate_pairs:
                used_idx.add(int(a))
                used_idx.add(int(b))
            keep_indices = sorted(used_idx)
            # rebuild pts to only include paired points and produce ordered img_pts where each pair's two points are consecutive
            new_pts = []
            pairs = []
            for (a, b) in candidate_pairs:
                pairs.append((int(a), int(b)))
                new_pts.append(pts[int(a)])
                new_pts.append(pts[int(b)])
            img_pts = np.array(new_pts, dtype=np.float64).reshape((-1, 2))

    return img_pts, pairs


def detect_platform_circle(frame):
    """Detect circular platform using Hough Circle Transform. Returns (x,y,r) or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=80, param2=30, minRadius=80, maxRadius=300)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        return int(x), int(y), int(r)
    return None


class PairDetector:
    """Stateful detector that maintains EMA smoothing for midpoints and blob points.
    Use process(frame) each loop to get smoothed img_pts and pairs.
    """
    def __init__(self, K, point_diameter_m=POINT_DIAMETER_M, working_distance_m=WORKING_DISTANCE_M,
                 alpha=POINT_SMOOTH_ALPHA, detector=None, max_pair_px=MAX_PAIR_PX,
                 pair_sep_m=PAIR_SEP_M, pair_sep_tol=PAIR_SEP_TOL):
        self.K = K
        self.point_diameter_m = point_diameter_m
        self.working_distance_m = working_distance_m
        self.alpha = float(alpha)
        self.max_pair_px = max_pair_px
        self.pair_sep_m = pair_sep_m
        self.pair_sep_tol = pair_sep_tol
        self.detector = detector
        self.smoothed_midpoints = None
        self.smoothed_blobs = None

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # create detector lazily if needed
        if self.detector is None:
            if (self.K is not None) and (hasattr(self.K, 'shape') and self.K.shape[0] >= 1):
                try:
                    est_area = estimate_pixel_area_for_size(self.point_diameter_m, self.K, self.working_distance_m)
                    min_area = int(max(5, est_area * 0.3))
                    max_area = int(max(20, est_area * 2.0))
                except Exception:
                    min_area, max_area = 50, 500
            else:
                min_area, max_area = 50, 500
            self.detector = make_blob_detector(min_area=min_area, max_area=max_area, dark=True)

        keypoints = self.detector.detect(blurred)

        # estimate expected diameter px if K provided
        if (self.K is not None) and (hasattr(self.K, 'shape') and self.K.shape[0] >= 1):
            fx = float(self.K[0, 0])
            expected_diameter_px = (self.point_diameter_m * fx) / self.working_distance_m
            min_diameter_px = expected_diameter_px * 0.5
            max_diameter_px = expected_diameter_px * 1.5
        else:
            min_diameter_px = 0
            max_diameter_px = 1e9

        filtered_keypoints = []
        for kp in keypoints:
            if min_diameter_px <= kp.size <= max_diameter_px:
                filtered_keypoints.append(kp)
        keypoints = filtered_keypoints

        if keypoints:
            refined_pts = [refine_centroid_subpixel(gray, kp.pt, win=11) for kp in keypoints]
            pts = np.array(refined_pts, dtype=np.float32)
        else:
            pts = np.empty((0, 2), dtype=np.float32)

        pairs = []
        img_pts = None
        if len(pts) >= 2:
            candidate_pairs = pair_points_by_distance(pts, expected_pairs=(len(pts) // 2), max_pair_px=self.max_pair_px)
            
            # Filter pairs by expected physical separation (e.g., 7cm between screws)
            if self.pair_sep_m is not None and len(candidate_pairs) > 0 and self.K is not None:
                fx = float(self.K[0, 0])
                expected_px = (self.pair_sep_m * fx) / self.working_distance_m
                tol_px = expected_px * self.pair_sep_tol
                filtered_pairs = []
                for a, b in candidate_pairs:
                    d = float(np.linalg.norm(pts[int(a)] - pts[int(b)]))
                    if abs(d - expected_px) <= tol_px:
                        filtered_pairs.append((a, b))
                candidate_pairs = filtered_pairs
            
            if candidate_pairs:
                midpts = []
                blobs = []
                for (a, b) in candidate_pairs:
                    mid = 0.5 * (pts[a] + pts[b])
                    midpts.append(mid)
                    blobs.append(pts[a])
                    blobs.append(pts[b])

                midpts_array = np.array(midpts, dtype=np.float64)
                blobs_array = np.array(blobs, dtype=np.float64).reshape((-1, 2))

                # smoothing (EMA)
                if self.smoothed_midpoints is None or self.smoothed_midpoints.shape[0] != midpts_array.shape[0]:
                    self.smoothed_midpoints = midpts_array.copy()
                else:
                    self.smoothed_midpoints = (1 - self.alpha) * self.smoothed_midpoints + self.alpha * midpts_array

                if self.smoothed_blobs is None or self.smoothed_blobs.shape[0] != blobs_array.shape[0]:
                    self.smoothed_blobs = blobs_array.copy()
                else:
                    self.smoothed_blobs = (1 - self.alpha) * self.smoothed_blobs + self.alpha * blobs_array

                img_pts = self.smoothed_blobs.copy()
                pairs = [(int(a), int(b)) for a, b in candidate_pairs]

        return img_pts, pairs


def detect_pairs_from_frame(frame, K, detector=None, point_diameter_m=POINT_DIAMETER_M,
                            working_distance_m=WORKING_DISTANCE_M, max_pair_px=MAX_PAIR_PX):
    """Backward-compatible wrapper: single-frame detection without smoothing (alpha=1.0)."""
    pd = PairDetector(K, point_diameter_m=point_diameter_m, working_distance_m=working_distance_m,
                      alpha=1.0, detector=detector, max_pair_px=max_pair_px)
    return pd.process(frame)


def make_blob_detector(min_area=50, max_area=500, min_circ=None, dark=True, min_size=5, max_size=30):
    """
    Create blob detector tuned for circular screw heads (~1cm diameter).
    min_area/max_area: pixel area range (more restrictive for 1cm points)
    min_circ: minimum circularity (higher = more circular), defaults to MIN_CIRCULARITY
    min_size/max_size: blob diameter range in pixels
    """
    if min_circ is None:
        min_circ = MIN_CIRCULARITY
    
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = float(min_area)
    params.maxArea = float(max_area)
    params.filterByCircularity = True
    params.minCircularity = float(min_circ)
    params.filterByInertia = True
    params.minInertiaRatio = 0.6  # More circular (increased from 0.5)
    params.filterByConvexity = True
    params.minConvexity = 0.85  # More convex/rounder (increased from 0.8)
    params.filterByColor = True
    params.blobColor = 0 if dark else 255
    return cv2.SimpleBlobDetector_create(params)


def refine_centroid_subpixel(gray, pt, win=11):
    """
    Refine a blob center to subpixel accuracy using a small local patch.
    Assumes dark blobs on lighter background (inverts patch).
    """
    h, w = gray.shape[:2]
    x0 = int(round(pt[0]))
    y0 = int(round(pt[1]))
    half = win // 2
    x1 = max(0, x0 - half)
    x2 = min(w, x0 + half + 1)
    y1 = max(0, y0 - half)
    y2 = min(h, y0 + half + 1)
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return np.array(pt, dtype=np.float32)
    # Make blobs bright by inverting local patch (dark blobs -> bright)
    vals = 255 - patch.astype(np.float32)
    # Simple adaptive threshold relative to mean; this keeps it robust
    thr = np.mean(vals) * 0.35
    mask = (vals > thr).astype(np.uint8)
    M = cv2.moments(mask, binaryImage=True)
    if M.get("m00", 0) != 0:
        cx = (M["m10"] / M["m00"]) + x1
        cy = (M["m01"] / M["m00"]) + y1
        return np.array([cx, cy], dtype=np.float32)
    # fallback to intensity-weighted centroid
    s = np.sum(vals)
    if s > 1e-6:
        cols = np.arange(x1, x2)
        rows = np.arange(y1, y2)
        cx = (cols * np.sum(vals, axis=0)).sum() / s
        cy = (rows * np.sum(vals, axis=1)).sum() / s
        return np.array([cx, cy], dtype=np.float32)
    return np.array(pt, dtype=np.float32)


def pair_points_by_distance(pts, expected_pairs=3, max_pair_px=None):
    """
    Greedy nearest pairing. Optionally discard pairs longer than max_pair_px.
    Only accept mutual nearest-neighbor pairs to avoid forming pairs with isolated stray points.
    pts: (N,2)
    returns list of index pairs [(i,j),...]
    """
    N = len(pts)
    if N < 2:
        return []
    # Distance matrix
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)

    # Nearest neighbor index for each point
    nn = np.argmin(D, axis=1)

    # Collect mutual nearest-neighbor pairs (i < j to avoid duplicates)
    mutual = []
    for i in range(N):
        j = int(nn[i])
        if j == i:
            continue
        if int(nn[j]) == i:
            a, b = (i, j) if i < j else (j, i)
            d = D[a, b]
            mutual.append((a, b, d))

    if not mutual:
        return []

    # Sort mutual pairs by distance (closest first) and pick up to expected_pairs, ensuring disjointness
    mutual.sort(key=lambda x: x[2])
    pairs = []
    used = set()
    for a, b, d in mutual:
        if len(pairs) >= expected_pairs:
            break
        if a in used or b in used:
            continue
        if (max_pair_px is not None) and (d > max_pair_px):
            continue
        pairs.append((a, b))
        used.add(a)
        used.add(b)

    return pairs



def main():
    ap = argparse.ArgumentParser(description="Live plane-normal estimation from 6 screws (3 pairs) on a circular plate.")
    ap.add_argument("--calib", type=str, default=None, help="npz with K, dist (optional)")
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--fullcal", action="store_true", help="Run full calibration GUI and save config.json then exit")
    args = ap.parse_args()

    # Load intrinsics if calibration file is provided
    K = None
    dist = None
    if args.calib is not None:
        try:
            data = np.load(args.calib, allow_pickle=True)
            K = data["K"].astype(np.float64)
            dist = data["dist"].astype(np.float64)
            print(f"[INFO] Loaded camera calibration from {args.calib}")
        except FileNotFoundError:
            print(f"[WARN] Calibration file {args.calib} not found. Running without camera calibration.")
        except Exception as e:
            print(f"[WARN] Failed to load calibration file: {e}. Running without camera calibration.")
    else:
        print("[INFO] No calibration file provided. Running without camera calibration.")

    # Build model points for 3 joint centers (z=0)
    thetas = np.radians(ANGLES_DEG)
    model_radius = 1.0
    obj_pts_3 = np.array([[model_radius * np.cos(t), model_radius * np.sin(t), 0.0] for t in thetas],
                         dtype=np.float64)

    # Estimate pixel area and diameter for the expected point size
    if K is not None:
        est_area = estimate_pixel_area_for_size(POINT_DIAMETER_M, K, WORKING_DISTANCE_M)
        fx = K[0, 0]
        expected_diameter_px = (POINT_DIAMETER_M * fx) / WORKING_DISTANCE_M
        min_diameter_px = expected_diameter_px * 0.7
        max_diameter_px = expected_diameter_px * 1.3
        # Use a range around the estimated area (±50% tolerance)
        min_area = int(est_area * 0.3)  # Allow some smaller
        max_area = int(est_area * 2.0)  # Allow some larger
        print(f"Point detection: {POINT_DIAMETER_M * 1000:.1f}mm diameter")
        print(f"  Estimated pixel diameter: {expected_diameter_px:.1f} px (range: {min_diameter_px:.1f}-{max_diameter_px:.1f} px)")
        print(f"  Estimated pixel area: {est_area:.1f} px (range: {min_area}-{max_area} px)")
        print(f"  Minimum circularity: {MIN_CIRCULARITY:.2f}")
        print(f"  Point smoothing alpha: {POINT_SMOOTH_ALPHA:.2f}")
        # Show pair separation constraint
        if PAIR_SEP_M is not None:
            expected_sep_px = (PAIR_SEP_M * fx) / WORKING_DISTANCE_M
            tol_px = expected_sep_px * PAIR_SEP_TOL
            print(f"Pair separation: {PAIR_SEP_M * 100:.1f}cm between screws")
            print(f"  Expected pixel separation: {expected_sep_px:.1f} px (±{tol_px:.1f} px)")
    else:
        # Use default values when no calibration is available
        min_area = 50
        max_area = 500
        print(f"Point detection: {POINT_DIAMETER_M * 1000:.1f}mm diameter")
        print(f"  Using default detection parameters (no calibration available)")
        print(f"  Point smoothing alpha: {POINT_SMOOTH_ALPHA:.2f}")

    # Detector tuned for ~1cm diameter dark, round points
    detector = make_blob_detector(min_area=min_area, max_area=max_area, min_circ=0.5, dark=True)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return

    print("Controls: q/ESC=quit, n=print pair info; script now only performs blob detection and pairing.")
    points_found = False

    # Point smoothing buffers
    # smoothed_midpoints: EMA for the 3 midpoints (for display)
    smoothed_midpoints = None
    # smoothed_blobs: EMA for the full 6 detected blob coordinates (order: for each pair, the two points in the same order as `pairs`)
    smoothed_blobs = None
    point_smooth_alpha = POINT_SMOOTH_ALPHA

    # --- Mouse / motor assignment state ---
    motor_assignments = {}  # map 'A'/'B'/'C' -> stored (x,y) coordinates (np.array)
    motor_order = ['A', 'B', 'C']
    next_motor_idx = 0
    cursor_pt = None
    # Latest detected midpoints and pairs (used by mouse callback)
    latest_pts = np.empty((0, 2), dtype=np.float32)
    latest_pairs = []
    # Per-motor smoothed midpoints (EMA in image pixels) and stability tracking
    motor_smoothed = {'A': None, 'B': None, 'C': None}
    motor_last_moved = {'A': None, 'B': None, 'C': None}
    motor_stable_since = {'A': None, 'B': None, 'C': None}
    motors_stable_flag = False
    # stability parameters
    STABLE_PIXEL_THRESH = 5   # px movement threshold
    STABLE_TIME_SEC = 1.0       # duration to consider stable
    smoothing_alpha = POINT_SMOOTH_ALPHA
    # Colors for motor highlights (B, G, R)
    motor_colors = {'A': (0, 0, 255), 'B': (255, 0, 0), 'C': (0, 255, 0)}

    # Mouse callback to track cursor and assign motors on left-click
    def on_mouse(event, x, y, flags, param):
        nonlocal cursor_pt, motor_assignments, next_motor_idx
        nonlocal latest_pts, latest_pairs
        if event == cv2.EVENT_MOUSEMOVE:
            cursor_pt = np.array([float(x), float(y)], dtype=np.float32)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Use latest detected midpoint pairs to select a pair nearest to the click
            if latest_pts is None or latest_pts.size == 0 or len(latest_pairs) == 0:
                print("No detected pairs to assign. Wait until the detector finds pairs.")
                return
            click = np.array([float(x), float(y)], dtype=np.float32)
            # compute distances to midpoints (latest_pts are the smoothed blob points (pairs ordered))
            # midpoints are every two entries (0/1 -> pair0 midpoint at index 0.5). We'll build midpoints list
            midpoints = []
            for (a, b) in latest_pairs:
                p = 0.5 * (latest_pts[a] + latest_pts[b])
                midpoints.append(p)
            midpoints = np.array(midpoints, dtype=np.float32)
            dists = np.linalg.norm(midpoints - click, axis=1)
            idx = int(np.argmin(dists))
            min_dist = float(dists[idx])
            if min_dist > 80.0:
                print(f"Click too far from nearest pair midpoint ({min_dist:.1f}px); ignore.")
                return
            # compute midpoint and candidate point
            a, b = latest_pairs[idx]
            candidate = 0.5 * (latest_pts[a] + latest_pts[b])
            # check if this midpoint (or a midpoint very near it) is already assigned
            for k, v in motor_assignments.items():
                if np.linalg.norm(v - candidate) < 12.0:
                    print(f"Midpoint already assigned to motor {k}")
                    return
            if next_motor_idx >= len(motor_order):
                print("All motors already assigned. Press 'r' to reset assignments.")
                return
            name = motor_order[next_motor_idx]
            motor_assignments[name] = candidate.copy()
            # initialize smoothing state for this motor
            motor_smoothed[name] = candidate.copy()
            motor_last_moved[name] = cv2.getTickCount() / cv2.getTickFrequency()
            motor_stable_since[name] = None
            next_motor_idx += 1
            print(f"Assigned motor {name} to blob at {candidate}")

    # create window and register mouse callback
    win_name = "plane normal (live)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse)

    # flag to ensure we only launch the full calibration UI once after computing the coordinate_frame
    launched_fullcal = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        disp = frame.copy()

        if PREVIEW and K is not None and dist is not None:
            newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (frame.shape[1], frame.shape[0]), 0)
            disp = cv2.undistort(frame, K, dist, None, newK)

        # --- Detect screws as dark blobs ---
        # Pre-blur to reduce sensor speckle that causes jitter
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        keypoints = detector.detect(blurred)

        # Filter by size (diameter) - estimate expected pixel diameter for point size
        if K is not None:
            fx = K[0, 0]
            expected_diameter_px = (POINT_DIAMETER_M * fx) / WORKING_DISTANCE_M
            min_diameter_px = expected_diameter_px * 0.5  # Allow 50% smaller
            max_diameter_px = expected_diameter_px * 1.5  # Allow 50% larger
        else:
            # Use permissive defaults when no calibration is available
            min_diameter_px = 5
            max_diameter_px = 50

        # Filter keypoints by size
        filtered_keypoints = []
        for kp in keypoints:
            if min_diameter_px <= kp.size <= max_diameter_px:
                filtered_keypoints.append(kp)
        keypoints = filtered_keypoints

        # refine each keypoint to subpixel centroid (use original gray for detail)
        if keypoints:
            refined_pts = [refine_centroid_subpixel(gray, kp.pt, win=11) for kp in keypoints]
            pts = np.array(refined_pts, dtype=np.float32)
        else:
            pts = np.empty((0, 2), dtype=np.float32)

        # Enforce pair property early: only keep blobs that belong to mutual nearest-neighbor pairs.
        # This removes stray single blobs that don't have a partner and reduces false positives.
        if len(pts) >= 2:
            # ask for as many disjoint mutual pairs as possible
            candidate_pairs = pair_points_by_distance(pts, expected_pairs=(len(pts) // 2), max_pair_px=MAX_PAIR_PX)
            if candidate_pairs:
                # optional: if physical pair separation is known, filter pairs by expected pixel separation
                if PAIR_SEP_M is not None and len(candidate_pairs) > 0 and K is not None:
                    fx = K[0, 0]
                    expected_px = (PAIR_SEP_M * fx) / WORKING_DISTANCE_M
                    tol_px = expected_px * PAIR_SEP_TOL
                    filtered_pairs = []
                    for a, b in candidate_pairs:
                        d = float(np.linalg.norm(pts[int(a)] - pts[int(b)]))
                        if abs(d - expected_px) <= tol_px:
                            filtered_pairs.append((a, b))
                    candidate_pairs = filtered_pairs

                if candidate_pairs:
                    used_idx = set()
                    for a, b in candidate_pairs:
                        used_idx.add(int(a))
                        used_idx.add(int(b))
                    # Rebuild keypoints and pts to only include points that are part of a mutual pair
                    keep_indices = sorted(used_idx)
                    # update keypoints from refined pts
                    keypoints = [cv2.KeyPoint(float(pts[i][0]), float(pts[i][1]), 1.0) for i in keep_indices]
                    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)

        for kp in keypoints:
            cv2.circle(disp, (int(kp.pt[0]), int(kp.pt[1])), max(2, int(kp.size / 2) if hasattr(kp, 'size') else 3), (0, 255, 255), 2)

        # Compute pairs and midpoints for however many mutual pairs we can find
        img_pts = None
        pairs = []
        if len(pts) >= 2:
            # Find as many disjoint mutual pairs as possible
            pairs = pair_points_by_distance(pts, expected_pairs=(len(pts) // 2), max_pair_px=MAX_PAIR_PX)
            if pairs:
                midpts = []
                blobs = []
                for (i, j) in pairs:
                    p = 0.5 * (pts[i] + pts[j])
                    midpts.append(p)
                    cv2.line(disp, tuple(pts[i].astype(int)), tuple(pts[j].astype(int)), (0, 200, 0), 2)
                    cv2.circle(disp, tuple(p.astype(int)), 6, (255, 0, 0), -1)
                    # keep the two blob points in the same order as the pair
                    blobs.append(pts[i])
                    blobs.append(pts[j])

                midpts_array = np.array(midpts, dtype=np.float64)
                blobs_array = np.array(blobs, dtype=np.float64).reshape((-1, 2))

                # Apply smoothing to midpoints (handle variable number of pairs)
                if smoothed_midpoints is None or smoothed_midpoints.shape[0] != midpts_array.shape[0]:
                    smoothed_midpoints = midpts_array.copy()
                else:
                    smoothed_midpoints = (1 - point_smooth_alpha) * smoothed_midpoints + point_smooth_alpha * midpts_array

                # Apply smoothing to the blob points (preserve ordering)
                if smoothed_blobs is None or smoothed_blobs.shape[0] != blobs_array.shape[0]:
                    smoothed_blobs = blobs_array.copy()
                else:
                    smoothed_blobs = (1 - point_smooth_alpha) * smoothed_blobs + point_smooth_alpha * blobs_array

                # For downstream use/drawing, provide smoothed image points for the available pairs
                img_pts = smoothed_blobs.copy()
                points_found = True
            else:
                points_found = False
                smoothed_midpoints = None  # Reset smoothing when pairs are lost
                smoothed_blobs = None
        else:
            points_found = False
            smoothed_midpoints = None  # Reset smoothing when not enough points
            smoothed_blobs = None

        # update latest detected points and pairs for mouse selection
        if 'pts' in locals() and pts is not None:
            latest_pts = pts.copy()
        else:
            latest_pts = np.empty((0, 2), dtype=np.float32)
        latest_pairs = pairs if 'pairs' in locals() else []

        # ---- Motor smoothing and stability detection ----
        t_now = cv2.getTickCount() / cv2.getTickFrequency()
        # Update each assigned motor by finding the closest midpoint in img_pts (if available)
        if 'img_pts' in locals() and img_pts is not None and len(img_pts) > 0:
            num_pairs = len(pairs)
            # construct midpoints from img_pts: each pair corresponds to two consecutive entries
            midpts = []
            for k in range(num_pairs):
                m = 0.5 * (img_pts[2 * k] + img_pts[2 * k + 1])
                midpts.append(m)
            midpts = np.array(midpts, dtype=np.float32)
        else:
            midpts = None

        # For each motor, update smoothing if we can find a corresponding midpoint
        for mname in motor_order:
            if mname not in motor_assignments:
                continue
            stored = motor_assignments[mname]
            found_pt = None
            if midpts is not None and midpts.size > 0:
                d = np.linalg.norm(midpts - stored.reshape((1, 2)), axis=1)
                j = int(np.argmin(d))
                if d[j] < 100.0:
                    found_pt = midpts[j]
            if found_pt is None:
                # fallback to stored assignment (no update)
                curr = stored.copy()
            else:
                curr = found_pt

            prev = motor_smoothed.get(mname)
            if prev is None:
                motor_smoothed[mname] = curr.copy()
                motor_last_moved[mname] = t_now
                motor_stable_since[mname] = None
            else:
                motor_smoothed[mname] = (1 - smoothing_alpha) * prev + smoothing_alpha * curr
                # movement since previous smoothed
                mv = np.linalg.norm(motor_smoothed[mname] - prev)
                if mv > STABLE_PIXEL_THRESH:
                    motor_last_moved[mname] = t_now
                    motor_stable_since[mname] = None
                else:
                    # if stable and not yet marked, set stable_since
                    if motor_stable_since[mname] is None:
                        motor_stable_since[mname] = t_now

        # Check if all motors are assigned and stable for required duration
        motors_assigned = all([m in motor_assignments for m in motor_order])
        motors_stable_flag = False
        if motors_assigned:
            all_stable = True
            for m in motor_order:
                ss = motor_stable_since.get(m)
                if ss is None or (t_now - ss) < STABLE_TIME_SEC:
                    all_stable = False
                    break
            if all_stable:
                motors_stable_flag = True
        # If stable and coordinate_frame not stored yet, compute it
        global coordinate_frame
        if motors_stable_flag and coordinate_frame is None:
            # Work directly in pixel coordinates
            pos_px = {}
            for m in motor_order:
                img = motor_smoothed[m]
                pos_px[m] = np.array([img[0], img[1]], dtype=np.float64)

            # centroid origin in pixels
            origin_px = (pos_px['A'] + pos_px['B'] + pos_px['C']) / 3.0
            # X axis toward Motor A from origin
            x_axis = pos_px['A'] - origin_px
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
            # provisional vector towards B, remove component along X to get Y in-plane
            v = pos_px['B'] - origin_px
            v = v - np.dot(v, x_axis) * x_axis
            y_axis = v / (np.linalg.norm(v) + 1e-12)

            coordinate_frame = {
                'origin_px': origin_px.tolist(),
                'x_axis': x_axis.tolist(),
                'y_axis': y_axis.tolist(),
                'timestamp': float(t_now)
            }
            print("Coordinate system stored (in pixels):")
            print(coordinate_frame)
            # Persist the computed coordinate_frame into config.json so other tools (or --fullcal)
            # can pick it up. Merge into existing config.json if present.
            try:
                cfg_path = "config.json"
                cfg = {}
                try:
                    with open(cfg_path, 'r') as f:
                        cfg = json.load(f)
                except FileNotFoundError:
                    cfg = {}
                except Exception:
                    # if file exists but can't be parsed, overwrite it
                    cfg = {}

                cfg['coordinate_frame'] = coordinate_frame
                with open(cfg_path, 'w') as f:
                    json.dump(cfg, f, indent=2)
                print(f"[SAVE] coordinate_frame written to {cfg_path}")
            except Exception as ex:
                print(f"[WARN] Failed to write coordinate_frame to config.json: {ex}")

            # Launch full calibration GUI once now that we have a valid coordinate frame.
            # Release the main capture and windows first to avoid camera conflicts.
            try:
                if not launched_fullcal:
                    launched_fullcal = True
                    try:
                        cap.release()
                    except Exception:
                        pass
                    try:
                        cv2.destroyAllWindows()
                    except Exception:
                        pass
                    print("[INFO] Launching full calibration UI now that coordinate_frame is available...")
                    calibrator = StewartPlatformCalibrator()
                    calibrator.CAM_INDEX = args.cam
                    calibrator.run()
                    # After calibrator finishes, exit the main program
                    return
            except Exception as ex:
                print(f"[WARN] Failed to write coordinate_frame to config.json: {ex}")

        # If we have any paired image points, compute and display pair separations and keep smoothed blobs
        if 'img_pts' in locals() and img_pts is not None:
            # compute pixel separations between paired blobs (pairs may be 1..3)
            num_pairs = len(pairs)
            pair_pixel_seps = [np.linalg.norm(img_pts[2 * k] - img_pts[2 * k + 1]) for k in range(num_pairs)]
            for k, sep in enumerate(pair_pixel_seps):
                cv2.putText(disp, f"Pair{k} sep: {sep:.1f}px", (12, 120 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            points_found = True
        else:
            # Points not found - show waiting message
            if len(pts) < 2:
                status_msg = f"Waiting for points... Found {len(pts)} blobs"
            else:
                status_msg = f"Waiting for valid pairs... Found {len(pts)} blobs and {len(pairs)} pairs"
            cv2.putText(disp, status_msg, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(disp, "Keep camera view steady", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Basic overlay counts
        cv2.putText(disp, f"Blobs: {len(pts)}", (12, disp.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw cursor position if available
        if cursor_pt is not None:
            cx, cy = int(cursor_pt[0]), int(cursor_pt[1])
            cv2.drawMarker(disp, (cx, cy), (200, 200, 200), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
            cv2.putText(disp, f"cursor: ({cx},{cy})", (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Draw motor highlights and labels for assigned blobs (persist across frames)
        if motor_assignments:
            for mname, stored in motor_assignments.items():
                # prefer smoothed motor positions for label placement
                label_pos = motor_smoothed.get(mname)
                if label_pos is None:
                    # fallback to stored coordinate or nearest img pt
                    if (img_pts is not None) and (len(img_pts) > 0):
                        d = np.linalg.norm(img_pts - stored.reshape((1, 2)), axis=1)
                        j = int(np.argmin(d))
                        label_pos = img_pts[j]
                    else:
                        label_pos = stored
                lx, ly = int(label_pos[0]), int(label_pos[1])
                color = motor_colors.get(mname, (0, 255, 255))
                # draw filled circle to highlight the assigned blob
                cv2.circle(disp, (lx, ly), 10, color, -1)
                # draw a thin black outline for contrast
                cv2.circle(disp, (lx, ly), 12, (0, 0, 0), 2)
                # draw the label slightly offset
                tx, ty = lx + 12, ly - 12
                # black shadow for readability
                cv2.putText(disp, f"Motor {mname}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(disp, f"Motor {mname}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # If coordinate frame available, draw X/Y axes in image
        if coordinate_frame is not None:
            origin_px = np.array(coordinate_frame['origin_px'], dtype=np.float64)
            x_axis = np.array(coordinate_frame['x_axis'], dtype=np.float64)
            y_axis = np.array(coordinate_frame['y_axis'], dtype=np.float64)
            # Choose axis length in pixels for visualization
            axis_len_px = 100
            p_o = origin_px
            p_x = origin_px + x_axis * axis_len_px
            p_y = origin_px + y_axis * axis_len_px
            try:
                uo, vo = int(round(p_o[0])), int(round(p_o[1]))
                ux, vx = int(round(p_x[0])), int(round(p_x[1]))
                uy, vy = int(round(p_y[0])), int(round(p_y[1]))
                # draw axes: X=red, Y=green
                cv2.arrowedLine(disp, (uo, vo), (ux, vx), (0, 0, 255), 3, tipLength=0.05)
                cv2.putText(disp, 'X', (ux + 4, vx + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.arrowedLine(disp, (uo, vo), (uy, vy), (0, 255, 0), 3, tipLength=0.05)
                cv2.putText(disp, 'Y', (uy + 4, vy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(disp, 'Coordinate system stored', (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            except Exception:
                # drawing failed for degenerate values; skip drawing
                pass

        cv2.imshow(win_name, disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord('n'):
            # Compute and print pair separations on-demand for the current frame (no persistent storage)
            if 'img_pts' in locals() and img_pts is not None:
                num_pairs = len(pairs)
                pair_pixel_seps = [np.linalg.norm(img_pts[2 * k] - img_pts[2 * k + 1]) for k in range(num_pairs)]
                print("---- Pair separations (px) ----")
                for idx, s in enumerate(pair_pixel_seps):
                    print(f"Pair {idx}: {s:.2f} px")
            else:
                print("No pairs detected to print separations.")
        if key == ord('r'):
            # reset motor assignments
            motor_assignments.clear()
            next_motor_idx = 0
            # reset smoothing/stability and coordinate frame
            for m in motor_smoothed.keys():
                motor_smoothed[m] = None
                motor_last_moved[m] = None
                motor_stable_since[m] = None
            motors_stable_flag = False
            coordinate_frame = None
            print("Motor assignments and coordinate frame reset. Re-select motors.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()