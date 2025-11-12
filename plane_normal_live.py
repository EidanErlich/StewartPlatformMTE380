#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import itertools
import json
from math import atan2, acos, degrees

# ---- FIXED PLATE GEOMETRY (your setup) ----
RADIUS_M = 0.15                  # 15 cm
ANGLES_DEG = [0.0, 120.0, 240.0]  # evenly spaced, CCW from +X


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


def solve_p3p_best(obj3, img3, K, dist):
    # Run P3P and pick the solution with minimum reprojection error
    obj3 = np.asarray(obj3)
    img3 = np.asarray(img3)
    npts = obj3.shape[0]

    # Ensure proper dtypes for OpenCV
    obj3_f = obj3.astype(np.float32)
    img3_f = img3.astype(np.float32)
    K_f = K.astype(np.float64)
    dist_f = dist.astype(np.float64)

    rvecs = []
    tvecs = []

    if npts >= 4:
        # Use generic multi-solution solver when 4+ points available
        try:
            ok, rvecs_out, tvecs_out, _ = cv2.solvePnPGeneric(obj3_f, img3_f, K_f, dist_f, flags=cv2.SOLVEPNP_ITERATIVE)
        except Exception:
            return None
        if not ok or len(rvecs_out) == 0:
            return None
        rvecs = rvecs_out
        tvecs = tvecs_out
    elif npts == 3:
        # Fallback: use solvePnP (P3P) when exactly 3 points are given.
        try:
            res = cv2.solvePnP(obj3_f, img3_f, K_f, dist_f, flags=cv2.SOLVEPNP_P3P)
        except Exception:
            return None
        # cv2.solvePnP can return either (retval, rvec, tvec) or (rvec, tvec) depending on OpenCV build
        if isinstance(res, tuple) and len(res) == 3:
            retval, rvec, tvec = res
            if not bool(retval):
                return None
        else:
            rvec, tvec = res
        rvecs = [rvec]
        tvecs = [tvec]
    else:
        return None

    best = None
    best_err = 1e9
    for rvec, tvec in zip(rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj3.astype(np.float32), rvec, tvec, K_f, dist_f)
        err = np.mean(np.linalg.norm(proj.reshape(-1, 2) - img3.astype(np.float32), axis=1))
        if err < best_err:
            best_err = err
            best = (rvec, tvec, best_err)
    return best  # (rvec, tvec, err)


def refine_pose(obj3, img3, K, dist, rvec, tvec):
    """Levenberg-Marquardt refinement for a more stable normal."""
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            obj3.astype(np.float32),
            img3.astype(np.float32),
            K, dist, rvec, tvec
        )
    except Exception:
        pass
    return rvec, tvec


def normal_from_rvec(rvec):
    R, _ = cv2.Rodrigues(rvec)
    n = (R @ np.array([0.0, 0.0, 1.0])).ravel()
    n /= np.linalg.norm(n)
    return n


def main():
    ap = argparse.ArgumentParser(description="Live plane-normal estimation from 6 screws (3 pairs) on a circular plate.")
    ap.add_argument("--calib", type=str, default="camera_calib.npz", help="npz with K, dist")
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--preview", action="store_true", help="undistort preview window")
    ap.add_argument("--json", action="store_true", help="print one-line JSON each frame (for downstream parsing)")
    ap.add_argument("--max_pair_px", type=float, default=None, help="optional max pixel distance to accept screw pairs")
    ap.add_argument("--point_diameter_m", type=float, default=0.01, help="Expected point diameter in meters (default: 0.01 = 1cm)")
    ap.add_argument("--working_distance_m", type=float, default=0.5, help="Estimated working distance in meters (default: 0.5m)")
    ap.add_argument("--point_smooth_alpha", type=float, default=0.3, help="Point smoothing factor (0-1, lower=more smoothing, default: 0.3)")
    ap.add_argument("--debug", action="store_true", help="Enable debug prints for P3P/pose estimation failures")
    ap.add_argument("--use_unit_model", action="store_true",
                    help="Use a unit-radius model for the three plate points (ignore RADIUS_M).\n"
                         "This makes the rotation/normal estimation depend only on the angular placement of the points, not the physical radius.")
    args = ap.parse_args()

    # Load intrinsics
    data = np.load(args.calib, allow_pickle=True)
    K = data["K"].astype(np.float64)
    dist = data["dist"].astype(np.float64)

    # Build model points for 3 joint centers (z=0)
    thetas = np.radians(ANGLES_DEG)
    model_radius = 1.0 if args.use_unit_model else RADIUS_M
    if args.use_unit_model:
        print("Using unit-radius model for pose estimation: normal will be independent of RADIUS_M")
    obj_pts_3 = np.array([[model_radius * np.cos(t), model_radius * np.sin(t), 0.0] for t in thetas],
                         dtype=np.float64)

    # Estimate pixel area and diameter for the expected point size
    est_area = estimate_pixel_area_for_size(args.point_diameter_m, K, args.working_distance_m)
    fx = K[0, 0]
    expected_diameter_px = (args.point_diameter_m * fx) / args.working_distance_m
    min_diameter_px = expected_diameter_px * 0.5
    max_diameter_px = expected_diameter_px * 1.5
    # Use a range around the estimated area (±50% tolerance)
    min_area = int(est_area * 0.3)  # Allow some smaller
    max_area = int(est_area * 2.0)  # Allow some larger
    print(f"Point detection: {args.point_diameter_m * 1000:.1f}mm diameter")
    print(f"  Estimated pixel diameter: {expected_diameter_px:.1f} px (range: {min_diameter_px:.1f}-{max_diameter_px:.1f} px)")
    print(f"  Estimated pixel area: {est_area:.1f} px (range: {min_area}-{max_area} px)")
    print(f"  Point smoothing alpha: {args.point_smooth_alpha:.2f}")

    # Detector tuned for ~1cm diameter dark, round points
    detector = make_blob_detector(min_area=min_area, max_area=max_area, min_circ=0.5, dark=True)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return

    print("Controls: q/ESC=quit, n=print normal/angles, (optional) --json for machine-readable stream.")
    alpha = 0.25  # smoothing for the normal
    normal_smoothed = None
    best_err = None
    points_found = False

    # Point smoothing buffers
    # smoothed_midpoints: EMA for the 3 midpoints (for display)
    smoothed_midpoints = None
    # smoothed_blobs: EMA for the full 6 detected blob coordinates (order: for each pair, the two points in the same order as `pairs`)
    smoothed_blobs = None
    point_smooth_alpha = args.point_smooth_alpha

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        disp = frame.copy()

        if args.preview:
            newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (frame.shape[1], frame.shape[0]), 0)
            disp = cv2.undistort(frame, K, dist, None, newK)

        # --- Detect screws as dark blobs ---
        keypoints = detector.detect(gray)

        # Filter by size (diameter) - estimate expected pixel diameter for 1cm
        fx = K[0, 0]
        expected_diameter_px = (args.point_diameter_m * fx) / args.working_distance_m
        min_diameter_px = expected_diameter_px * 0.5  # Allow 50% smaller
        max_diameter_px = expected_diameter_px * 1.5  # Allow 50% larger

        # Filter keypoints by size
        filtered_keypoints = []
        for kp in keypoints:
            if min_diameter_px <= kp.size <= max_diameter_px:
                filtered_keypoints.append(kp)
        keypoints = filtered_keypoints

        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32) if keypoints else np.empty((0, 2), dtype=np.float32)

        # Enforce pair property early: only keep blobs that belong to mutual nearest-neighbor pairs.
        # This removes stray single blobs that don't have a partner and reduces false positives.
        if len(pts) >= 2:
            # ask for as many disjoint mutual pairs as possible
            candidate_pairs = pair_points_by_distance(pts, expected_pairs=(len(pts) // 2), max_pair_px=args.max_pair_px)
            if candidate_pairs:
                used_idx = set()
                for a, b in candidate_pairs:
                    used_idx.add(int(a))
                    used_idx.add(int(b))
                # Rebuild keypoints and pts to only include points that are part of a mutual pair
                keep_indices = sorted(used_idx)
                keypoints = [keypoints[i] for i in keep_indices]
                pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)

        for kp in keypoints:
            cv2.circle(disp, (int(kp.pt[0]), int(kp.pt[1])), max(2, int(kp.size / 2)), (0, 255, 255), 2)

        img_pts_3 = None
        pairs = []
        if len(pts) >= 6:
            # Pair into 3 nearest pairs; optional distance limit to avoid cross-pairing
            pairs = pair_points_by_distance(pts, expected_pairs=3, max_pair_px=args.max_pair_px)
            if len(pairs) == 3:
                midpts = []
                for (i, j) in pairs:
                    p = 0.5 * (pts[i] + pts[j])
                    midpts.append(p)
                    cv2.line(disp, tuple(pts[i].astype(int)), tuple(pts[j].astype(int)), (0, 200, 0), 2)
                    cv2.circle(disp, tuple(p.astype(int)), 6, (255, 0, 0), -1)
                midpts_array = np.array(midpts, dtype=np.float64)

                # Build the flat list of 6 image points (two per pair) in the same pair order
                blobs6 = []
                pair_pixel_seps = []
                for (i, j) in pairs:
                    p1 = pts[i]
                    p2 = pts[j]
                    blobs6.append(p1)
                    blobs6.append(p2)
                    pair_pixel_seps.append(np.linalg.norm(p1 - p2))
                blobs6_array = np.array(blobs6, dtype=np.float64).reshape((6, 2))

                # Apply smoothing to midpoints (for display)
                if smoothed_midpoints is None:
                    smoothed_midpoints = midpts_array.copy()
                else:
                    smoothed_midpoints = (1 - point_smooth_alpha) * smoothed_midpoints + point_smooth_alpha * midpts_array

                # Apply smoothing to the 6 blob points (preserve ordering)
                if smoothed_blobs is None:
                    smoothed_blobs = blobs6_array.copy()
                else:
                    smoothed_blobs = (1 - point_smooth_alpha) * smoothed_blobs + point_smooth_alpha * blobs6_array

                # Draw smoothed midpoints in a different color to show smoothing effect
                for smp in smoothed_midpoints:
                    cv2.circle(disp, tuple(smp.astype(int)), 4, (0, 255, 255), -1)  # Yellow filled circle for smoothed

                # For pose estimation, use the full 6 points (smoothed if available)
                img_pts_6 = smoothed_blobs.copy()
                points_found = True
            else:
                points_found = False
                smoothed_midpoints = None  # Reset smoothing when pairs are lost
                smoothed_blobs = None
        else:
            points_found = False
            smoothed_midpoints = None  # Reset smoothing when not enough points
            smoothed_blobs = None

        normal = None
        inc = None
        azi = None

        if 'img_pts_6' in locals() and img_pts_6 is not None:
            # We now try to use all 6 detected blobs (two per model center) for pose.
            # For each model center we estimate the physical pair separation from pixel separation and working distance,
            # then build the 6 object points (center +/- half_sep along the local tangent) and try all mappings
            # from the detected pairs to the model centers as well as the two internal orderings per pair.
            best = None
            best_err_local = 1e9
            best_perm = None

            # Compute pair separations in meters using simple pinhole approximation
            fx = K[0, 0]
            # pair_pixel_seps were computed when pairs were built; recompute here for safety
            pair_pixel_seps = []
            for idx in range(3):
                pA = img_pts_6[2 * idx]
                pB = img_pts_6[2 * idx + 1]
                pair_pixel_seps.append(np.linalg.norm(pA - pB))

            pair_seps_m = [ (px * args.working_distance_m) / fx for px in pair_pixel_seps ]

            # Build candidate object points for each model center: two points offset along the tangent
            obj_pts_pairs = []  # list of (pt_plus, pt_minus) for each center
            for k, t in enumerate(thetas):
                center = obj_pts_3[k]
                # tangent unit vector in model plane (CCW around +Z)
                tan = np.array([-np.sin(t), np.cos(t), 0.0], dtype=np.float64)
                half = pair_seps_m[k] / 2.0
                obj_plus = center + half * tan
                obj_minus = center - half * tan
                obj_pts_pairs.append((obj_plus, obj_minus))

            # Try permutations mapping detected pair indices to model center indices
            for perm in itertools.permutations(range(3)):
                # For each of the 3 pairs, each has two possible internal orderings (which blob maps to + vs -)
                for order_bits in range(8):
                    obj6 = []
                    img6 = []
                    valid = True
                    for model_idx in range(3):
                        pair_idx = perm[model_idx]
                        # image points for this pair in the same order they were stored (2*pair_idx, 2*pair_idx+1)
                        im_a = img_pts_6[2 * pair_idx]
                        im_b = img_pts_6[2 * pair_idx + 1]
                        # choose assignment based on bit
                        bit = (order_bits >> model_idx) & 1
                        if bit == 0:
                            obj_a = obj_pts_pairs[model_idx][0]
                            obj_b = obj_pts_pairs[model_idx][1]
                            img_a = im_a
                            img_b = im_b
                        else:
                            obj_a = obj_pts_pairs[model_idx][1]
                            obj_b = obj_pts_pairs[model_idx][0]
                            img_a = im_a
                            img_b = im_b

                        obj6.append(obj_a)
                        obj6.append(obj_b)
                        img6.append(img_a)
                        img6.append(img_b)

                    obj6 = np.array(obj6, dtype=np.float64)
                    img6 = np.array(img6, dtype=np.float64)

                    hyp = solve_p3p_best(obj6, img6, K, dist)
                    if hyp is None:
                        if args.debug:
                            print(f"P6P failed for perm {perm} order {order_bits}")
                        continue
                    rvec, tvec, err = hyp

                    # refine
                    rvec, tvec = refine_pose(obj6.astype(np.float32), img6.astype(np.float32), K, dist, rvec, tvec)

                    proj, _ = cv2.projectPoints(obj6, rvec, tvec, K, dist)
                    err_ref = np.mean(np.linalg.norm(proj.reshape(-1, 2) - img6, axis=1))

                    if err_ref < best_err_local:
                        best = (rvec, tvec)
                        best_err_local = err_ref
                        best_perm = (perm, order_bits)

            if best is not None:
                # Update outer best_err with the successful solution
                best_err = best_err_local
                rvec, tvec = best
                n = normal_from_rvec(rvec)

                # Make normal face the camera (optional convention)
                cam_to_plate = -tvec.reshape(-1)
                if np.dot(n, cam_to_plate) < 0:
                    n *= -1

                # Smooth
                if normal_smoothed is None:
                    normal_smoothed = n.copy()
                else:
                    normal_smoothed = (1 - alpha) * normal_smoothed + alpha * n
                    normal_smoothed /= np.linalg.norm(normal_smoothed)

                # Angles
                inc = acos(np.clip(normal_smoothed[2], -1.0, 1.0))          # radians
                azi = atan2(normal_smoothed[1], normal_smoothed[0])         # radians

                # ----- Overlay: projected plate and axes -----
                try:
                    # Project a sampled circle (plate rim) into the image and draw a translucent fill
                    circ_pts_3d = np.array([[model_radius * np.cos(tt), model_radius * np.sin(tt), 0.0]
                                            for tt in np.linspace(0, 2 * np.pi, 48, endpoint=False)], dtype=np.float64)
                    proj_circ, _ = cv2.projectPoints(circ_pts_3d, rvec, tvec, K, dist)
                    proj_circ_pts = proj_circ.reshape(-1, 2).astype(int)

                    overlay = disp.copy()
                    # fill the projected plate region with a translucent color (orange)
                    cv2.fillPoly(overlay, [proj_circ_pts], color=(0, 128, 255))
                    cv2.addWeighted(overlay, 0.15, disp, 0.85, 0, disp)

                    # Draw plate rim outline
                    cv2.polylines(disp, [proj_circ_pts], isClosed=True, color=(0, 128, 255), thickness=2)

                    # Draw coordinate axes at the plate center (origin = [0,0,0] in model frame)
                    axis_len = max(0.05, model_radius * 0.5)  # meters
                    axes3d = np.array([[0.0, 0.0, 0.0],
                                       [axis_len, 0.0, 0.0],
                                       [0.0, axis_len, 0.0],
                                       [0.0, 0.0, axis_len]], dtype=np.float64)
                    proj_axes, _ = cv2.projectPoints(axes3d, rvec, tvec, K, dist)
                    pa = proj_axes.reshape(-1, 2)
                    origin_px = tuple(pa[0].astype(int))
                    x_px = tuple(pa[1].astype(int))
                    y_px = tuple(pa[2].astype(int))
                    z_px = tuple(pa[3].astype(int))

                    # X = red, Y = green, Z = blue
                    cv2.line(disp, origin_px, x_px, (0, 0, 255), 2)
                    cv2.line(disp, origin_px, y_px, (0, 255, 0), 2)
                    cv2.line(disp, origin_px, z_px, (255, 0, 0), 2)
                    # small circles at endpoints
                    cv2.circle(disp, x_px, 4, (0, 0, 255), -1)
                    cv2.circle(disp, y_px, 4, (0, 255, 0), -1)
                    cv2.circle(disp, z_px, 4, (255, 0, 0), -1)
                except Exception:
                    # If projection fails for any reason, continue without overlays
                    pass

                # HUD
                cv2.putText(disp, f"Reproj err: {best_err:.3f}px", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(disp, f"Normal: [{normal_smoothed[0]:+.3f}, {normal_smoothed[1]:+.3f}, {normal_smoothed[2]:+.3f}]",
                            (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(disp, f"Inclination: {degrees(inc):.2f} deg  Azimuth: {degrees(azi):.2f} deg",
                            (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if best_perm is not None:
                    cv2.putText(disp, f"Pairing perm: {best_perm}", (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

                # Machine-friendly stream (one line per frame)
                if args.json:
                    out = {
                        "normal_cam": [float(normal_smoothed[0]), float(normal_smoothed[1]), float(normal_smoothed[2])],
                        "inclination_deg": float(degrees(inc)),
                        "azimuth_deg": float(degrees(azi)),
                        "reproj_err_px": float(best_err)
                    }
                    print(json.dumps(out), flush=True)
            else:
                # Points detected but pose estimation failed - show debug info
                cv2.putText(disp, "Pose estimation failed - check calibration", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(disp, f"Found {len(pairs)} pairs but P3P solving failed", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # Reset best_err to None when no solution found
                best_err = None
                if args.debug:
                    print("--- DEBUG: Pose estimation failed ---")
                    print("obj_pts_3=\n", obj_pts_3)
                    # Print the 6 image points used for PnP
                    try:
                        print("img_pts_6=\n", img_pts_6)
                    except Exception:
                        print("img_pts_6 not available")
                    print("pairs=", pairs)
                    try:
                        # compute midpoints from the 6 image points and show their pairwise distances
                        mid_from_img = np.array([0.5 * (img_pts_6[2 * k] + img_pts_6[2 * k + 1]) for k in range(3)])
                        Dimg = np.linalg.norm(mid_from_img[:, None, :] - mid_from_img[None, :, :], axis=2)
                        print("Image midpoints distance matrix:\n", Dimg)
                    except Exception:
                        pass
                    # Try fallback: use solvePnPGeneric on the 3 model centers and measured midpoints
                    try:
                        obj3_f = obj_pts_3.astype(np.float32)
                        img3_f = np.array([0.5 * (img_pts_6[2 * k] + img_pts_6[2 * k + 1]) for k in range(3)]).astype(np.float32)
                        K_f = K.astype(np.float64)
                        dist_f = dist.astype(np.float64)
                        ok, rvecs_out, tvecs_out, inliers = cv2.solvePnPGeneric(obj3_f, img3_f, K_f, dist_f, flags=cv2.SOLVEPNP_P3P)
                        print("solvePnPGeneric (P3P) ok:", ok)
                        if ok:
                            for idx, (rv, tv) in enumerate(zip(rvecs_out, tvecs_out)):
                                proj, _ = cv2.projectPoints(obj_pts_3.astype(np.float32), rv, tv, K_f, dist_f)
                                err = np.mean(np.linalg.norm(proj.reshape(-1, 2) - img3_f, axis=1))
                                print(f"  Solution {idx}: reproj err = {err}")
                        else:
                            print("  solvePnPGeneric did not return solutions")
                    except Exception as e:
                        print("  Fallback solvePnPGeneric exception:", e)
        else:
            # Points not found - show waiting message
            if len(pts) < 6:
                status_msg = f"Waiting for points... Found {len(pts)}/6 blobs"
            else:
                status_msg = f"Waiting for valid pairs... Found {len(pts)} blobs but {len(pairs)}/3 pairs"
            cv2.putText(disp, status_msg, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(disp, "Keep camera view steady", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Basic overlay counts
        cv2.putText(disp, f"Blobs: {len(pts)}", (12, disp.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("plane normal (live)", disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord('n') and normal_smoothed is not None and inc is not None and azi is not None and best_err is not None:
            print("---- Normal / Angles ----")
            print("normal (camera):", normal_smoothed)
            print("inclination (deg):", degrees(inc))
            print("azimuth (deg):    ", degrees(azi))
            print("reproj err (px):  ", best_err)
        elif key == ord('n'):
            print("Waiting for points to be detected before normal can be calculated.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()