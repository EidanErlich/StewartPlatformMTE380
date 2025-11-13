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
    # Colors for motor highlights (B, G, R)
    motor_colors = {'A': (0, 0, 255), 'B': (255, 0, 0), 'C': (0, 255, 0)}

    # Mouse callback to track cursor and assign motors on left-click
    def on_mouse(event, x, y, flags, param):
        nonlocal cursor_pt, motor_assignments, next_motor_idx
        if event == cv2.EVENT_MOUSEMOVE:
            cursor_pt = np.array([float(x), float(y)], dtype=np.float32)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Safely check for detected points in this scope
            if 'pts' not in locals() or pts is None or len(pts) == 0:
                print("No detected blobs to assign.")
                return
            click = np.array([float(x), float(y)], dtype=np.float32)
            # find nearest detected blob
            dists = np.linalg.norm(pts - click, axis=1)
            idx = int(np.argmin(dists))
            min_dist = float(dists[idx])
            # threshold: ignore clicks that are clearly far from blobs
            if min_dist > 50.0:
                print(f"Click too far from nearest blob ({min_dist:.1f}px); ignore.")
                return
            candidate = pts[idx]
            # check if this blob (or a blob very near it) is already assigned
            for k, v in motor_assignments.items():
                if np.linalg.norm(v - candidate) < 8.0:
                    print(f"Blob already assigned to motor {k}")
                    return
            if next_motor_idx >= len(motor_order):
                print("All motors already assigned. Press 'r' to reset assignments.")
                return
            name = motor_order[next_motor_idx]
            motor_assignments[name] = candidate.copy()
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
                # if we have current detected points, try to find the closest one to the stored coordinate
                label_pos = None
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
            print("Motor assignments reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()