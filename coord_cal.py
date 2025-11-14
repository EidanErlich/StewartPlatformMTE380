#!/usr/bin/env python3
import cv2
import numpy as np
import argparse

# ---- FIXED PLATE GEOMETRY (your setup) ----
ANGLES_DEG = [0.0, 120.0, 240.0]  # evenly spaced, CCW from +X

# ---- RUNTIME CONFIG (hardcoded, not CLI) ----
# Only --calib and --cam are accepted on the command line; everything else is fixed here.
PREVIEW = False                 # undistort preview window
JSON_OUTPUT = False             # print one-line JSON each frame
MAX_PAIR_PX = None              # optional max pixel distance to accept screw pairs
POINT_DIAMETER_M = 0.01        # expected point diameter in meters (1cm)
WORKING_DISTANCE_M = 0.5       # estimated working distance in meters
POINT_SMOOTH_ALPHA = 0.3       # point smoothing factor (0-1)
PAIR_SEP_M = None              # known physical separation (meters) between two screws in a pair
PAIR_SEP_TOL = 0.4             # tolerance fraction for PAIR_SEP_M when filtering pairs

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


def load_camera_calib(calib_path="camera_calib.npz"):
    """Load camera intrinsics from an npz file."""
    data = np.load(calib_path, allow_pickle=True)
    K = data["K"].astype(np.float64)
    dist = data["dist"].astype(np.float64)
    return K, dist


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
        detector = make_blob_detector(min_area=min_area, max_area=max_area, min_circ=0.5, dark=True)

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
                 alpha=POINT_SMOOTH_ALPHA, detector=None, max_pair_px=MAX_PAIR_PX):
        self.K = K
        self.point_diameter_m = point_diameter_m
        self.working_distance_m = working_distance_m
        self.alpha = float(alpha)
        self.max_pair_px = max_pair_px
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
            self.detector = make_blob_detector(min_area=min_area, max_area=max_area, min_circ=0.5, dark=True)

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


def make_blob_detector(min_area=50, max_area=500, min_circ=0.5, dark=True, min_size=5, max_size=30):
    """
    Create blob detector tuned for ~1cm diameter points.
    min_area/max_area: pixel area range (more restrictive for 1cm points)
    min_circ: minimum circularity (higher = more circular)
    min_size/max_size: blob diameter range in pixels
    """
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = float(min_area)
    params.maxArea = float(max_area)
    params.filterByCircularity = True
    params.minCircularity = float(min_circ)
    params.filterByInertia = True
    params.minInertiaRatio = 0.5  # More circular
    params.filterByConvexity = True
    params.minConvexity = 0.8  # More convex (rounder)
    params.filterByColor = True
    params.blobColor = 0 if dark else 255
    # Filter by size (diameter)
    params.filterByArea = True  # Already set, but ensure it's on
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
    ap.add_argument("--calib", type=str, default="camera_calib.npz", help="npz with K, dist")
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    args = ap.parse_args()

    # Load intrinsics
    data = np.load(args.calib, allow_pickle=True)
    K = data["K"].astype(np.float64)
    dist = data["dist"].astype(np.float64)

    # Build model points for 3 joint centers (z=0)
    thetas = np.radians(ANGLES_DEG)
    model_radius = 1.0
    obj_pts_3 = np.array([[model_radius * np.cos(t), model_radius * np.sin(t), 0.0] for t in thetas],
                         dtype=np.float64)

    # Estimate pixel area and diameter for the expected point size
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

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        disp = frame.copy()

        if PREVIEW:
            newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (frame.shape[1], frame.shape[0]), 0)
            disp = cv2.undistort(frame, K, dist, None, newK)

        # --- Detect screws as dark blobs ---
        # Pre-blur to reduce sensor speckle that causes jitter
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        keypoints = detector.detect(blurred)

        # Filter by size (diameter) - estimate expected pixel diameter for point size
        fx = K[0, 0]
        expected_diameter_px = (POINT_DIAMETER_M * fx) / WORKING_DISTANCE_M
        min_diameter_px = expected_diameter_px * 0.5  # Allow 50% smaller
        max_diameter_px = expected_diameter_px * 1.5  # Allow 50% larger

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
                if PAIR_SEP_M is not None and len(candidate_pairs) > 0:
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
            # Backproject smoothed image points to approximate 3D using WORKING_DISTANCE_M
            fx = K[0, 0]
            fy = K[1, 1] if K.shape[0] > 1 else K[0, 0]
            cx = K[0, 2]
            cy = K[1, 2]
            Z = WORKING_DISTANCE_M
            pos3 = {}
            for m in motor_order:
                img = motor_smoothed[m]
                x = (img[0] - cx) * Z / fx
                y = (img[1] - cy) * Z / fy
                z = Z
                pos3[m] = np.array([x, y, z], dtype=np.float64)

            # centroid origin
            origin = (pos3['A'] + pos3['B'] + pos3['C']) / 3.0
            # X axis toward Motor A from origin
            x_axis = pos3['A'] - origin
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
            # provisional vector towards B, remove component along X to get Y in-plane
            v = pos3['B'] - origin
            v = v - np.dot(v, x_axis) * x_axis
            y_axis = v / (np.linalg.norm(v) + 1e-12)
            # Z is perpendicular to plane
            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)
            # ensure z_axis points toward the camera (camera at origin); flip if needed
            to_cam = -origin
            if np.dot(z_axis, to_cam) < 0:
                z_axis = -z_axis

            coordinate_frame = {
                'origin_m': origin.tolist(),
                'x_axis': x_axis.tolist(),
                'y_axis': y_axis.tolist(),
                'z_axis': z_axis.tolist(),
                'timestamp': float(t_now)
            }
            print("Coordinate system stored:")
            print(coordinate_frame)

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

        # If coordinate frame available, draw X/Y/Z arrows projected into image
        if coordinate_frame is not None:
            # small helper to project camera-frame 3D point to image pixel
            def project_pt(pt_cam):
                Xc = pt_cam[0]
                Yc = pt_cam[1]
                Zc = pt_cam[2]
                if Zc == 0:
                    Zc = 1e-6
                u = (K[0, 0] * Xc) / Zc + K[0, 2]
                v = (K[1, 1] * Yc) / Zc + K[1, 2]
                return int(round(u)), int(round(v))

            origin = np.array(coordinate_frame['origin_m'], dtype=np.float64)
            x_axis = np.array(coordinate_frame['x_axis'], dtype=np.float64)
            y_axis = np.array(coordinate_frame['y_axis'], dtype=np.float64)
            z_axis = np.array(coordinate_frame['z_axis'], dtype=np.float64)
            # Choose axis length in meters for visualization
            axis_len_m = 0.1
            p_o = origin
            p_x = origin + x_axis * axis_len_m
            p_y = origin + y_axis * axis_len_m
            p_z = origin + z_axis * axis_len_m
            try:
                uo, vo = project_pt(p_o)
                ux, vx = project_pt(p_x)
                uy, vy = project_pt(p_y)
                uz, vz = project_pt(p_z)
                # draw axes: X=red, Y=green, Z=blue
                cv2.arrowedLine(disp, (uo, vo), (ux, vx), (0, 0, 255), 3, tipLength=0.05)
                cv2.putText(disp, 'X', (ux + 4, vx + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.arrowedLine(disp, (uo, vo), (uy, vy), (0, 255, 0), 3, tipLength=0.05)
                cv2.putText(disp, 'Y', (uy + 4, vy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.arrowedLine(disp, (uo, vo), (uz, vz), (255, 0, 0), 3, tipLength=0.05)
                cv2.putText(disp, 'Z', (uz + 4, vz + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(disp, 'Coordinate system stored', (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            except Exception:
                # projection failed for degenerate values; skip drawing
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